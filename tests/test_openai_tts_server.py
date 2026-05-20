import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import torch
from fastapi.testclient import TestClient

from omnivoice.openai_tts_server import (
    DEFAULT_AUDIO_CHUNK_THRESHOLD,
    ChunkSynthesisError,
    ChunkSynthesisResult,
    SecondaryWorkerManager,
    SpeechRequest,
    TextChunk,
    TextSanitizationOptions,
    VOICE_LOOKUP,
    _build_text_chunks,
    _detect_sentences,
    _has_secondary_worker_gpu_headroom,
    _iter_ordered_chunk_results,
    _iter_unique_local_voice_prompt_specs,
    _plan_sentence_chunks_with_source,
    _prepare_request,
    _process_chunk_with_retry,
    _resolve_voice,
    _should_use_parallel_chunking,
    _split_long_chunk_by_budget,
    _synthesize_chunk_local,
    _supported_models,
    _supported_voices,
    app,
    sanitize_prompt_text,
    sanitize_speech_text,
    secondary_worker_manager,
    service,
)


class _FakeModel:
    def __init__(self) -> None:
        self.sampling_rate = 24000
        self.calls: list[dict[str, object]] = []
        self.voice_prompt_calls: list[dict[str, object]] = []

    def create_voice_clone_prompt(self, *, ref_audio: str, ref_text=None):
        prompt = {"ref_audio": ref_audio, "ref_text": ref_text}
        self.voice_prompt_calls.append(prompt)
        return prompt

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [torch.zeros(1, 480)]


class OpenAITTSServerTests(unittest.TestCase):
    def setUp(self) -> None:
        service._voice_prompt_cache.clear()

    def test_sanitize_speech_text_preserves_bracket_tags_and_removes_control_tokens(
        self,
    ) -> None:
        text = " Hello\tworld <|text_end|>\n[laughter] me@example.com "
        sanitized = sanitize_speech_text(
            text,
            language="en",
            options=TextSanitizationOptions(),
        )

        self.assertNotIn("<|text_end|>", sanitized)
        self.assertIn("[laughter]", sanitized)
        self.assertIn("me at example dot com", sanitized)
        self.assertEqual(sanitized, sanitized.strip())
        self.assertTrue(sanitized.endswith("."))

    def test_sanitize_prompt_text_strips_model_tokens(self) -> None:
        sanitized = sanitize_prompt_text(" <|lang_start|>  male, low pitch \n")
        self.assertEqual(sanitized, "male, low pitch")

    def test_sanitize_speech_text_applies_kokoro_style_normalization(self) -> None:
        sanitized = sanitize_speech_text(
            "Dr. Smith meets me at 10:05 pm. It costs $50.30 for 5km in 1998(s).",
            language="en",
            options=TextSanitizationOptions(unit_normalization=True),
        )

        self.assertIn("Doctor Smith", sanitized)
        self.assertIn("ten oh five pm", sanitized)
        self.assertIn("fifty dollars and thirty cents", sanitized)
        self.assertIn("five kilometers", sanitized)
        self.assertIn("nineteen ninety-eight", sanitized)

    def test_supported_models_and_voices_include_openwebui_facing_entries(self) -> None:
        model_ids = {model["id"] for model in _supported_models()}
        voice_ids = {voice["id"] for voice in _supported_voices()}

        self.assertIn("tts-1", model_ids)
        self.assertIn("gpt-4o-mini-tts", model_ids)
        self.assertIn("alloy", voice_ids)
        self.assertIn("british_man", voice_ids)

    def test_unique_local_voice_prompt_specs_are_deduplicated(self) -> None:
        specs = _iter_unique_local_voice_prompt_specs()
        self.assertTrue(specs)
        self.assertEqual(len(specs), len({cache_key for cache_key, _, _ in specs}))

    def test_prewarm_local_voice_prompts_populates_cache_once(self) -> None:
        fake_model = _FakeModel()
        specs = [
            ("voice-a", Path("/tmp/voice-a.wav"), "alpha"),
            ("voice-b", Path("/tmp/voice-b.wav"), "beta"),
        ]

        with patch(
            "omnivoice.openai_tts_server._iter_unique_local_voice_prompt_specs",
            return_value=specs,
        ):
            warmed_first = service._prewarm_local_voice_prompts_sync(fake_model)
            warmed_second = service._prewarm_local_voice_prompts_sync(fake_model)

        self.assertEqual(warmed_first, 2)
        self.assertEqual(warmed_second, 0)
        self.assertEqual(len(fake_model.voice_prompt_calls), 2)

    def test_resolve_voice_prefers_local_reference_when_available(self) -> None:
        resolved = _resolve_voice("alloy")
        option = VOICE_LOOKUP["alloy"]

        self.assertEqual(resolved.voice_id, "alloy")
        if option.has_local_sample():
            self.assertIsNotNone(resolved.ref_audio_path)
            self.assertIsNone(resolved.instruct)
            self.assertTrue((resolved.ref_text or "").strip())
        else:
            self.assertIsNone(resolved.ref_audio_path)
            self.assertIsNotNone(resolved.instruct)

    def test_audio_endpoint_forces_sentence_chunking_for_long_input(self) -> None:
        fake_model = _FakeModel()
        long_text = " ".join(f"Sentence {idx}." for idx in range(1, 60))

        with (
            patch.object(service, "get_model", new=AsyncMock(return_value=fake_model)),
            patch.object(
                secondary_worker_manager,
                "ensure_started",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "omnivoice.openai_tts_server._waveform_to_bytes",
                return_value=(b"audio-bytes", "audio/mpeg"),
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": long_text,
                    "voice": "alloy",
                    "response_format": "mp3",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"audio-bytes")
        self.assertGreater(int(response.headers["x-omnivoice-text-chunks"]), 1)
        self.assertEqual(response.headers["x-omnivoice-forced-chunking"], "true")

        call = fake_model.calls[0]
        generation_config = call["generation_config"]
        self.assertEqual(generation_config.audio_chunk_threshold, 0.0)
        self.assertTrue(call["text"].endswith("."))
        if VOICE_LOOKUP["alloy"].has_local_sample():
            self.assertIn("voice_clone_prompt", call)
            self.assertTrue(call["voice_clone_prompt"]["ref_text"].strip())
        else:
            self.assertIn("instruct", call)

    def test_audio_endpoint_keeps_default_chunk_threshold_for_short_input(self) -> None:
        fake_model = _FakeModel()

        with (
            patch.object(service, "get_model", new=AsyncMock(return_value=fake_model)),
            patch(
                "omnivoice.openai_tts_server._waveform_to_bytes",
                return_value=(b"audio-bytes", "audio/mpeg"),
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": "Hello world",
                    "voice": "alloy",
                    "response_format": "mp3",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-omnivoice-text-chunks"], "1")
        self.assertEqual(response.headers["x-omnivoice-forced-chunking"], "false")

        call = fake_model.calls[0]
        generation_config = call["generation_config"]
        self.assertEqual(generation_config.audio_chunk_threshold, DEFAULT_AUDIO_CHUNK_THRESHOLD)
        if VOICE_LOOKUP["alloy"].has_local_sample():
            self.assertIn("voice_clone_prompt", call)
            self.assertTrue(call["voice_clone_prompt"]["ref_text"].strip())

    def test_audio_endpoint_allows_ref_text_override_for_local_voice(self) -> None:
        fake_model = _FakeModel()

        with (
            patch.object(service, "get_model", new=AsyncMock(return_value=fake_model)),
            patch(
                "omnivoice.openai_tts_server._waveform_to_bytes",
                return_value=(b"audio-bytes", "audio/mpeg"),
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": "Hello world",
                    "voice": "alloy",
                    "ref_text": "Custom reference sentence.",
                    "response_format": "mp3",
                },
            )

        self.assertEqual(response.status_code, 200)
        if VOICE_LOOKUP["alloy"].has_local_sample():
            self.assertEqual(
                fake_model.calls[0]["voice_clone_prompt"]["ref_text"],
                "Custom reference sentence.",
            )

    def test_audio_endpoint_validates_speed_bounds(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": "Hello world",
                    "voice": "alloy",
                    "speed": 4.5,
                },
            )

        self.assertEqual(response.status_code, 422)

    def test_audio_endpoint_uses_upstream_default_generation_params(self) -> None:
        """Verify the server builds a config matching upstream OmniVoice defaults.

        Regression guard: ensures the server never silently uses wrong generation
        parameters that caused gibberish output before the mask fix.
        """
        fake_model = _FakeModel()

        with (
            patch.object(service, "get_model", new=AsyncMock(return_value=fake_model)),
            patch(
                "omnivoice.openai_tts_server._waveform_to_bytes",
                return_value=(b"audio-bytes", "audio/mpeg"),
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/audio/speech",
                json={"input": "Hello world", "voice": "alloy", "response_format": "mp3"},
            )

        self.assertEqual(response.status_code, 200)
        gen = fake_model.calls[0]["generation_config"]
        self.assertEqual(gen.num_step, 32)
        self.assertAlmostEqual(gen.guidance_scale, 2.0)
        self.assertAlmostEqual(gen.t_shift, 0.1)
        self.assertAlmostEqual(gen.layer_penalty_factor, 5.0)
        self.assertAlmostEqual(gen.position_temperature, 5.0)
        self.assertAlmostEqual(gen.class_temperature, 0.0)
        self.assertTrue(gen.denoise)
        self.assertTrue(gen.postprocess_output)

    def test_frontend_page_shows_credits_and_voice_summary(self) -> None:
        with TestClient(app) as client:
            response = client.get("/ui")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Thanks to the original OmniVoice creators", response.text)
        self.assertIn("OpenAI-compatible TTS", response.text)
        self.assertIn("british_man", response.text)
        self.assertIn("cordobes_man", response.text)


class ChunkPlannerTests(unittest.TestCase):
    def test_sentence_detector_handles_abbreviations_decimals_ellipses_and_multilingual_punctuation(
        self,
    ) -> None:
        text = "Dr. Smith measured 3.14 cm. Wait... really? Hola! 你好。"
        with (
            patch("omnivoice.openai_tts_server._split_sentences_pysbd", return_value=None),
            patch("omnivoice.openai_tts_server._split_sentences_nltk", return_value=None),
        ):
            sentences, source = _detect_sentences(text)

        self.assertEqual(source, "regex")
        self.assertEqual(
            sentences,
            ["Dr. Smith measured 3.14 cm.", "Wait...", "really?", "Hola!", "你好。"],
        )

    def test_empty_chunks_are_removed_before_deterministic_ids_are_assigned(self) -> None:
        chunks = _build_text_chunks(["First.", " ", "", "Second.", "Third."])

        self.assertEqual([chunk.chunk_id for chunk in chunks], [1, 2, 3])
        self.assertEqual([chunk.worker_id for chunk in chunks], [1, 2, 1])
        self.assertEqual([chunk.text for chunk in chunks], ["First.", "Second.", "Third."])

    def test_long_chunks_split_by_character_budget(self) -> None:
        parts = _split_long_chunk_by_budget("alpha beta gamma delta epsilon", 12)

        self.assertTrue(all(len(part) <= 12 for part in parts))
        self.assertEqual(" ".join(parts), "alpha beta gamma delta epsilon")

    def test_adaptive_planner_merges_short_sentences_and_caps_queue(self) -> None:
        text = "Hi. Ok. This sentence is long enough to carry the first two."
        with (
            patch("omnivoice.openai_tts_server._split_sentences_pysbd", return_value=None),
            patch("omnivoice.openai_tts_server._split_sentences_nltk", return_value=None),
        ):
            chunks, source = _plan_sentence_chunks_with_source(text, min_chars=32)

        self.assertEqual(source, "regex")
        self.assertEqual(chunks, [text])


class ChunkedPipelineTests(unittest.IsolatedAsyncioTestCase):
    def _prepared_and_payload(self) -> tuple[object, SpeechRequest]:
        payload = SpeechRequest(
            input="One sentence. Two sentence. Three sentence.",
            voice="alloy",
            response_format="wav",
            sentence_chunking_min_chars=64,
        )
        return _prepare_request(payload), payload

    async def test_out_of_order_completion_streams_in_original_order(self) -> None:
        prepared, payload = self._prepared_and_payload()
        chunks = _build_text_chunks(["One.", "Two.", "Three."])
        delays = {1: 0.05, 2: 0.01, 3: 0.02}

        async def fake_run(worker_id, chunk, prepared, payload, *, request_id, retry_count):
            await asyncio.sleep(delays[chunk.chunk_id])
            now = asyncio.get_running_loop().time()
            return ChunkSynthesisResult(
                chunk_id=chunk.chunk_id,
                worker_id=worker_id,
                input_length=len(chunk.text),
                waveform=torch.full((1, 1), float(chunk.chunk_id)),
                sample_rate=24000,
                retry_count=retry_count,
                started_at=now,
                ended_at=now,
                latency_s=0.0,
            )

        seen: list[int] = []
        with patch("omnivoice.openai_tts_server._run_chunk_on_worker", new=fake_run):
            async for result in _iter_ordered_chunk_results(
                chunks,
                prepared,
                payload,
                request_id="test",
                secondary_available=True,
            ):
                seen.append(result.chunk_id)

        self.assertEqual(seen, [1, 2, 3])

    async def test_failed_chunk_retries_once_on_other_worker(self) -> None:
        prepared, payload = self._prepared_and_payload()
        chunk = TextChunk(chunk_id=1, text="One.", worker_id=1)
        calls: list[tuple[int, int]] = []

        async def fake_run(worker_id, chunk, prepared, payload, *, request_id, retry_count):
            calls.append((worker_id, retry_count))
            if retry_count == 0:
                raise RuntimeError("worker crashed")
            now = asyncio.get_running_loop().time()
            return ChunkSynthesisResult(
                chunk_id=chunk.chunk_id,
                worker_id=worker_id,
                input_length=len(chunk.text),
                waveform=torch.zeros(1, 1),
                sample_rate=24000,
                retry_count=retry_count,
                started_at=now,
                ended_at=now,
                latency_s=0.0,
            )

        with patch("omnivoice.openai_tts_server._run_chunk_on_worker", new=fake_run):
            result = await _process_chunk_with_retry(
                chunk,
                prepared,
                payload,
                request_id="test",
                secondary_available=True,
            )

        self.assertEqual(calls, [(1, 0), (2, 1)])
        self.assertEqual(result.worker_id, 2)
        self.assertEqual(result.retry_count, 1)

    async def test_worker_crash_propagates_clear_error_after_retry(self) -> None:
        prepared, payload = self._prepared_and_payload()
        chunk = TextChunk(chunk_id=2, text="Two.", worker_id=2)

        async def fake_run(worker_id, chunk, prepared, payload, *, request_id, retry_count):
            raise RuntimeError("boom")

        with patch("omnivoice.openai_tts_server._run_chunk_on_worker", new=fake_run):
            with self.assertRaises(ChunkSynthesisError) as caught:
                await _process_chunk_with_retry(
                    chunk,
                    prepared,
                    payload,
                    request_id="test",
                    secondary_available=True,
                )

        self.assertIn("Chunk 2 failed", str(caught.exception))
        self.assertIn("boom", str(caught.exception))

    async def test_cancellation_mid_stream_cancels_queued_tasks(self) -> None:
        prepared, payload = self._prepared_and_payload()
        chunks = _build_text_chunks(["One.", "Two.", "Three."])
        started = asyncio.Event()

        async def fake_run(worker_id, chunk, prepared, payload, *, request_id, retry_count):
            started.set()
            await asyncio.sleep(60)

        async def collect() -> list[int]:
            result_ids: list[int] = []
            async for result in _iter_ordered_chunk_results(
                chunks,
                prepared,
                payload,
                request_id="test",
                secondary_available=True,
            ):
                result_ids.append(result.chunk_id)
            return result_ids

        with patch("omnivoice.openai_tts_server._run_chunk_on_worker", new=fake_run):
            task = asyncio.create_task(collect())
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_concurrent_ensure_started_launches_only_one_secondary_worker(self) -> None:
        manager = SecondaryWorkerManager(port=59999)
        manager.set_lock(asyncio.Lock())
        launched = 0
        healthy = False

        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        def fake_start():
            nonlocal launched, healthy
            launched += 1
            healthy = True
            return FakeProcess()

        def fake_healthy():
            return healthy

        with (
            patch("omnivoice.openai_tts_server._has_secondary_worker_gpu_headroom", return_value=(True, "ok")),
            patch.object(manager, "_start_process_sync", side_effect=fake_start),
            patch.object(manager, "_wait_until_healthy", new=AsyncMock(return_value=None)),
            patch.object(manager, "is_healthy", side_effect=fake_healthy),
        ):
            results = await asyncio.gather(*(manager.ensure_started() for _ in range(8)))

        self.assertTrue(all(results))
        self.assertEqual(launched, 1)

    async def test_unhealthy_running_secondary_worker_is_terminated_before_restart(
        self,
    ) -> None:
        manager = SecondaryWorkerManager(port=59999)
        manager.set_lock(asyncio.Lock())

        class FakeProcess:
            def __init__(self, pid: int) -> None:
                self.pid = pid
                self.returncode = None
                self.terminated = False
                self.killed = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = 0

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.killed = True
                self.returncode = -9

        old_process = FakeProcess(111)
        new_process = FakeProcess(222)
        manager.process = old_process

        with (
            patch(
                "omnivoice.openai_tts_server._has_secondary_worker_gpu_headroom",
                return_value=(True, "ok"),
            ),
            patch.object(manager, "is_healthy", return_value=False),
            patch.object(manager, "_start_process_sync", return_value=new_process),
            patch.object(manager, "_wait_until_healthy", new=AsyncMock(return_value=None)),
        ):
            result = await manager.ensure_started()

        self.assertTrue(result)
        self.assertTrue(old_process.terminated)
        self.assertFalse(old_process.killed)
        self.assertIs(manager.process, new_process)
        self.assertEqual(manager.launch_count, 1)

    def test_gpu_fallback_path_reports_no_cuda_headroom(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            available, reason = _has_secondary_worker_gpu_headroom("cuda:0")

        self.assertFalse(available)
        self.assertIn("unavailable", reason.lower())

    async def test_local_chunk_synthesis_uses_chunk_local_threshold(self) -> None:
        payload = SpeechRequest(
            input="One sentence. Two sentence.",
            voice="alloy",
            response_format="wav",
        )
        prepared = _prepare_request(payload)
        prepared.chunk_plan = ["One sentence.", "Two sentence."]
        prepared.force_sentence_chunking = True
        prepared.generation_config.audio_chunk_threshold = 0.0
        chunk = TextChunk(chunk_id=1, text="One sentence.", worker_id=1)

        captured: dict[str, object] = {}

        async def fake_synthesize_waveform(
            chunk_prepared,
            payload_arg,
            *,
            request_id=None,
            worker_id=1,
        ):
            captured["prepared"] = chunk_prepared
            captured["payload"] = payload_arg
            captured["request_id"] = request_id
            captured["worker_id"] = worker_id
            return torch.zeros(1, 1), 24000

        with patch(
            "omnivoice.openai_tts_server._synthesize_prepared_waveform",
            new=fake_synthesize_waveform,
        ):
            result = await _synthesize_chunk_local(
                chunk,
                prepared,
                payload,
                request_id="test",
                retry_count=0,
            )

        chunk_prepared = captured["prepared"]
        self.assertEqual(chunk_prepared.text, chunk.text)
        self.assertEqual(chunk_prepared.chunk_plan, [chunk.text])
        self.assertFalse(chunk_prepared.force_sentence_chunking)
        self.assertEqual(
            chunk_prepared.generation_config.audio_chunk_threshold,
            DEFAULT_AUDIO_CHUNK_THRESHOLD,
        )
        self.assertEqual(result.worker_id, 1)

    def test_parallel_chunking_requires_beneficial_size(self) -> None:
        payload = SpeechRequest(
            input="One sentence. Two sentence. Three sentence. Four sentence.",
            voice="alloy",
            response_format="wav",
            sentence_chunking_min_chars=64,
        )
        prepared = _prepare_request(payload)
        prepared.chunk_plan = ["One sentence.", "Two sentence."]
        prepared.force_sentence_chunking = True

        with (
            patch("omnivoice.openai_tts_server.WORKER_MODE", False),
            patch("omnivoice.openai_tts_server.CHUNKED_PIPELINE_ENABLED", True),
            patch("omnivoice.openai_tts_server.CHUNK_MAX_WORKERS", 2),
            patch("omnivoice.openai_tts_server.CHUNK_MIN_PARALLEL_CHUNKS", 2),
            patch(
                "omnivoice.openai_tts_server.CHUNK_MIN_PARALLEL_CHARS",
                len(prepared.text) + 1,
            ),
        ):
            self.assertFalse(_should_use_parallel_chunking(prepared, payload))

        with (
            patch("omnivoice.openai_tts_server.WORKER_MODE", False),
            patch("omnivoice.openai_tts_server.CHUNKED_PIPELINE_ENABLED", True),
            patch("omnivoice.openai_tts_server.CHUNK_MAX_WORKERS", 2),
            patch("omnivoice.openai_tts_server.CHUNK_MIN_PARALLEL_CHUNKS", 2),
            patch("omnivoice.openai_tts_server.CHUNK_MIN_PARALLEL_CHARS", 1),
        ):
            self.assertTrue(_should_use_parallel_chunking(prepared, payload))


if __name__ == "__main__":
    unittest.main()
