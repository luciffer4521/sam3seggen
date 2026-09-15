import io
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from pipeline import PipelineOptions
from serve_api import blank_as_none, parse_optional_float, parse_optional_int
import serve_api


class BlankAsNoneTest(unittest.TestCase):
    def test_swagger_placeholders_become_none(self):
        for value in (None, "", "string", "integer", "number", "null"):
            self.assertIsNone(blank_as_none(value), value)

    def test_real_values_pass_through(self):
        self.assertEqual(blank_as_none("300"), "300")
        self.assertEqual(blank_as_none(0.5), 0.5)
        self.assertEqual(blank_as_none("body"), "body")

    def test_optional_numbers_parse_after_placeholders(self):
        self.assertIsNone(parse_optional_int("string"))
        self.assertEqual(parse_optional_int("300"), 300)
        self.assertIsNone(parse_optional_float(""))
        self.assertEqual(parse_optional_float("0.5"), 0.5)


class ConceptBankPlaceholderTest(unittest.TestCase):
    def test_swagger_string_keeps_the_deployed_bank(self):
        options = PipelineOptions.from_mapping({"concept_bank": "string"})
        self.assertEqual(options.concept_bank, PipelineOptions().concept_bank)


class SegmentFormTest(unittest.TestCase):
    def test_swagger_numeric_placeholders_do_not_422(self):
        captured = {}

        def fake_job(job_id, upload, filename, options, legacy=False):
            captured.update(options)
            return {"job_id": job_id, "parts": [], "seconds": 0}

        original = serve_api._run_job
        serve_api._run_job = fake_job
        try:
            client = TestClient(serve_api.app)
            response = client.post(
                "/segment",
                files={"glb": ("toy.glb", io.BytesIO(b"glb"), "model/gltf-binary")},
                data={
                    "prompts": "head, body, legs",
                    "unassigned_to": "string",
                    "min_atom_faces": "string",
                    "min_unit_faces": "string",
                    "color_tol": "string",
                    "min_recall": "string",
                    "concept_bank": "string",
                },
            )
        finally:
            serve_api._run_job = original
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(captured.get("min_atom_faces"))
        self.assertIsNone(captured.get("color_tol"))
        self.assertEqual(captured.get("unassigned_to"), "body")

    def test_glb_sent_as_text_returns_422_not_500(self):
        client = TestClient(serve_api.app)
        response = client.post(
            "/segment",
            data={"glb": "not-a-file\x00bytes", "prompts": "head"},
        )
        self.assertEqual(response.status_code, 422, response.text)
        body = response.json()
        detail = body["detail"]
        self.assertTrue(any(item.get("loc", [])[-1:] == ["glb"] for item in detail))
        self.assertTrue(
            any("file upload" in item.get("msg", "") for item in detail),
            detail,
        )


class JobLookupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old_jobs = serve_api.JOBS_DIR
        serve_api.JOBS_DIR = self.tmp
        self.old_current = serve_api._current["job_id"]
        serve_api._current["job_id"] = None

    def tearDown(self):
        serve_api.JOBS_DIR = self.old_jobs
        serve_api._current["job_id"] = self.old_current
        for root, dirs, files in os.walk(self.tmp, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.tmp)

    def _job(self, job_id, *, complete=False, parts=True):
        path = os.path.join(self.tmp, job_id)
        os.makedirs(os.path.join(path, "complete"), exist_ok=True)
        if parts:
            open(os.path.join(path, "parts.glb"), "wb").write(b"glb")
        if complete:
            open(os.path.join(path, "complete", "xpart_parts.glb"), "wb").write(b"glb")
        return path

    def test_list_and_latest_recover_job_id(self):
        job = "a" * 32
        self._job(job, complete=True)
        client = TestClient(serve_api.app)
        listing = client.get("/jobs")
        self.assertEqual(listing.status_code, 200, listing.text)
        body = listing.json()
        self.assertEqual(body["latest_job"], job)
        self.assertEqual(body["jobs"][0]["job_id"], job)
        self.assertEqual(body["jobs"][0]["state"], "done")
        latest = client.get("/jobs/latest")
        self.assertEqual(latest.status_code, 200, latest.text)
        self.assertEqual(latest.json()["job_id"], job)
        self.assertIn("complete", latest.json()["links"])
        health = client.get("/health")
        self.assertEqual(health.json()["latest_job"], job)

    def test_latest_empty_is_404(self):
        client = TestClient(serve_api.app)
        self.assertEqual(client.get("/jobs/latest").status_code, 404)
        self.assertEqual(client.get("/jobs").json()["jobs"], [])
