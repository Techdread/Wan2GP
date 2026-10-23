"""Regression checks for static upstream family discovery."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import agent_api_introspect as introspect
from agent_api_introspect import _extract_family_handlers_from_source


class FamilyDiscoveryTests(unittest.TestCase):
    def test_initial_list_and_appended_families(self):
        source = '''
family_handlers = ["models.wan.wan_handler"]
family_handlers.append("models.flux3.flux3_handler")
family_handlers.append("models.fibo2.fibo2_handler")
family_handlers.extend(["models.TTS.fish_audio.fish_audio_handler"])
'''
        self.assertEqual(_extract_family_handlers_from_source(source), [
            "models.wan.wan_handler", "models.flux3.flux3_handler",
            "models.fibo2.fibo2_handler", "models.TTS.fish_audio.fish_audio_handler",
        ])

    def test_dynamic_and_nested_calls_are_not_executed(self):
        source = '''
raise RuntimeError("must not run")
family_handlers = ["models.safe", dynamic_value]
family_handlers.append(load_plugin())
for plugin in plugins:
    family_handlers.append("models.dynamic")
def configure():
    family_handlers.append("models.nested")
'''
        self.assertEqual(_extract_family_handlers_from_source(source), ["models.safe"])

    def test_duplicates_are_removed_in_registration_order(self):
        source = '''
family_handlers = ["models.a", "models.b"]
family_handlers.append("models.a")
family_handlers.extend(("models.b", "models.c"))
'''
        self.assertEqual(_extract_family_handlers_from_source(source), ["models.a", "models.b", "models.c"])

    def test_missing_registration_returns_empty(self):
        self.assertEqual(_extract_family_handlers_from_source("x = []"), [])

    def test_mesh_outputs_are_not_listed_as_image_generators(self):
        self.assertEqual(introspect._capability_from_def(
            {"mesh_outputs": True, "image_outputs": True}, "trellis2"
        ), "3d-generation")

    def test_mesh_flag_precedes_video_name_heuristics(self):
        self.assertEqual(introspect._capability_from_def(
            {"mesh_outputs": True}, "hunyuan3d_21"
        ), "3d-generation")

    def test_static_supported_types_skip_runtime_bookkeeping(self):
        handler = SimpleNamespace(query_supported_types=Mock(side_effect=RuntimeError("must not migrate")))
        with patch.object(introspect, "_install_wgp_stub"), \
                patch.object(introspect, "_extract_family_handlers_from_wgp", return_value=["models.test"]), \
                patch.object(introspect, "_ast_extract_supported_types", return_value=["test"]), \
                patch.object(introspect.importlib, "import_module", return_value=SimpleNamespace(family_handler=handler)):
            handlers, errors = introspect._load_handlers()
        self.assertIs(handlers["test"], handler)
        self.assertEqual(errors, [])
        handler.query_supported_types.assert_not_called()

    def test_dynamic_supported_types_still_call_handler(self):
        handler = SimpleNamespace(query_supported_types=Mock(return_value=["dynamic"]))
        with patch.object(introspect, "_install_wgp_stub"), \
                patch.object(introspect, "_extract_family_handlers_from_wgp", return_value=["models.test"]), \
                patch.object(introspect, "_ast_extract_supported_types", return_value=[]), \
                patch.object(introspect.importlib, "import_module", return_value=SimpleNamespace(family_handler=handler)):
            handlers, errors = introspect._load_handlers()
        self.assertIs(handlers["dynamic"], handler)
        self.assertEqual(errors, [])
        handler.query_supported_types.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
