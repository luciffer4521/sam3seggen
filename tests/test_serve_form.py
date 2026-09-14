import io
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
