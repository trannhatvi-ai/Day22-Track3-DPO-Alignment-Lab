import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class NotebookIntegrityTests(unittest.TestCase):
    def test_sft_dataset_default_is_available_and_schema_is_supported(self):
        source = (ROOT / "notebooks" / "01_sft_mini.py").read_text(encoding="utf-8")

        self.assertIn("5CD-AI/Vietnamese-alpaca-gpt4-gg-translated", source)
        self.assertIn("instruction_vi", source)
        self.assertIn("output_vi", source)
        self.assertIn("ensure_qwen_chat_template(tokenizer)", source)

    def test_all_chat_template_call_sites_have_qwen_fallback(self):
        paths = [
            "notebooks/01_sft_mini.py",
            "notebooks/02_preference_data.py",
            "notebooks/04_compare_and_eval.py",
            "notebooks/06_benchmark.py",
            "scripts/prepare_preference_data.py",
            "colab/Lab22_DPO_T4.ipynb",
            "colab/Lab22_DPO_BigGPU.ipynb",
        ]
        for rel_path in paths:
            source = (ROOT / rel_path).read_text(encoding="utf-8")
            self.assertIn("apply_chat_template", source)
            self.assertIn("QWEN_CHAT_TEMPLATE", source, f"{rel_path} should define a fallback ChatML template")
            self.assertIn("ensure_qwen_chat_template(tokenizer)", source, f"{rel_path} should set tokenizer.chat_template")

    def test_colab_install_cells_quote_version_constraints(self):
        expected_specs = [
            '"unsloth>=2025.10,<2026.5"',
            '"transformers>=4.46,<5.0"',
            '"trl>=0.12,<0.20"',
            '"peft>=0.13,<1.0"',
            '"bitsandbytes>=0.44,<1.0"',
            '"lm-eval[ifeval,math]>=0.4.5,<1.0"',
        ]

        for rel_path in ["colab/Lab22_DPO_T4.ipynb", "colab/Lab22_DPO_BigGPU.ipynb"]:
            nb = json.loads((ROOT / rel_path).read_text(encoding="utf-8"))
            install_cells = [
                "".join(cell.get("source", []))
                for cell in nb["cells"]
                if cell.get("cell_type") == "code" and "!pip install" in "".join(cell.get("source", []))
            ]
            self.assertTrue(install_cells, f"{rel_path} has no install cell")
            install_source = "\n".join(install_cells)
            for spec in expected_specs:
                self.assertIn(spec, install_source, f"{rel_path} missing quoted spec {spec}")

            self.assertIsNone(
                re.search(r"(?<![\"'])\b(?:unsloth|trl|peft|bitsandbytes)>=", install_source),
                f"{rel_path} has an unquoted pip version spec that bash can treat as redirect",
            )

    def test_colab_early_gpu_probe_does_not_depend_on_later_notebook_state(self):
        for rel_path in ["colab/Lab22_DPO_T4.ipynb", "colab/Lab22_DPO_BigGPU.ipynb"]:
            nb = json.loads((ROOT / rel_path).read_text(encoding="utf-8"))
            probe_cells = [
                "".join(cell.get("source", []))
                for cell in nb["cells"]
                if "Enable GPU runtime" in "".join(cell.get("source", []))
            ]
            self.assertEqual(len(probe_cells), 1, f"{rel_path} should have one early GPU probe")
            self.assertNotIn("01-setup-gpu.png", probe_cells[0])
            self.assertNotIn("REPO_ROOT", probe_cells[0])
            self.assertNotIn("BASE_MODEL", probe_cells[0])

    def test_colab_sft_model_load_defines_chat_template_before_use(self):
        for rel_path in ["colab/Lab22_DPO_T4.ipynb", "colab/Lab22_DPO_BigGPU.ipynb"]:
            nb = json.loads((ROOT / rel_path).read_text(encoding="utf-8"))
            model_load_cells = [
                "".join(cell.get("source", []))
                for cell in nb["cells"]
                if "FastLanguageModel.from_pretrained" in "".join(cell.get("source", []))
                and "Qwen tokenizers ship without pad token" in "".join(cell.get("source", []))
            ]
            self.assertEqual(len(model_load_cells), 1, f"{rel_path} should have one SFT model-load cell")
            source = model_load_cells[0]
            self.assertLess(source.index("QWEN_CHAT_TEMPLATE"), source.index("ensure_qwen_chat_template(tokenizer)"))
            self.assertNotIn('ensure_qwen_chat_template(tokenizer)\n    print("Set tokenizer.pad_token', source)

    def test_notebooks_auto_write_required_screenshots(self):
        expected = {
            "notebooks/01_sft_mini.py": ["01-setup-gpu.png", "02-sft-loss.png"],
            "notebooks/03_dpo_train.py": ["03-dpo-reward-curves.png"],
            "notebooks/04_compare_and_eval.py": [
                "04-side-by-side-table.png",
                "05-manual-rubric.png",
            ],
            "notebooks/05_merge_deploy_gguf.py": ["06-gguf-smoke.png"],
            "notebooks/06_benchmark.py": ["07-benchmark-comparison.png"],
        }

        for rel_path, filenames in expected.items():
            source = (ROOT / rel_path).read_text(encoding="utf-8")
            for filename in filenames:
                self.assertIn(filename, source, f"{rel_path} should write {filename}")

    def test_trainers_use_processing_class_for_current_trl(self):
        paths = [
            "notebooks/01_sft_mini.py",
            "notebooks/03_dpo_train.py",
            "scripts/train_dpo.py",
            "colab/Lab22_DPO_T4.ipynb",
            "colab/Lab22_DPO_BigGPU.ipynb",
        ]
        for rel_path in paths:
            source = (ROOT / rel_path).read_text(encoding="utf-8")
            self.assertIn("processing_class=tokenizer", source, f"{rel_path} should pass tokenizer via processing_class")
            self.assertNotIn("tokenizer=tokenizer", source, f"{rel_path} still uses deprecated tokenizer=tokenizer")

    def test_dpo_adapter_is_continued_from_sft_and_deployed_directly(self):
        nb3_paths = [
            "notebooks/03_dpo_train.py",
            "scripts/train_dpo.py",
            "colab/Lab22_DPO_T4.ipynb",
            "colab/Lab22_DPO_BigGPU.ipynb",
        ]
        for rel_path in nb3_paths:
            source = (ROOT / rel_path).read_text(encoding="utf-8")
            if rel_path.startswith("colab/"):
                source = source.split("Stage from `notebooks/03_dpo_train.py`", 1)[1]
                source = source.split("Stage from `notebooks/04_compare_and_eval.py`", 1)[0]
            self.assertIn("PeftModel.from_pretrained", source)
            self.assertIn("is_trainable=True", source)
            self.assertNotIn("FastLanguageModel.get_peft_model(\n    model", source)

        nb5_paths = [
            "notebooks/05_merge_deploy_gguf.py",
            "colab/Lab22_DPO_T4.ipynb",
            "colab/Lab22_DPO_BigGPU.ipynb",
        ]
        for rel_path in nb5_paths:
            source = (ROOT / rel_path).read_text(encoding="utf-8")
            if rel_path.startswith("colab/"):
                source = source.split("Stage from `notebooks/05_merge_deploy_gguf.py`", 1)[1]
                source = source.split("Stage from `notebooks/06_benchmark.py`", 1)[0]
            self.assertIn("PeftModel.from_pretrained(model, str(DPO_PATH))", source)
            self.assertNotIn("PeftModel.from_pretrained(model, str(SFT_PATH))", source)

    def test_dpo_training_forces_eager_attention_for_t4_xformers_compatibility(self):
        for rel_path in [
            "notebooks/03_dpo_train.py",
            "scripts/train_dpo.py",
            "colab/Lab22_DPO_T4.ipynb",
            "colab/Lab22_DPO_BigGPU.ipynb",
        ]:
            path = ROOT / rel_path
            if path.suffix == ".ipynb":
                nb = json.loads(path.read_text(encoding="utf-8"))
                source = "\n".join(
                    "".join(cell.get("source", [])) for cell in nb["cells"]
                )
            else:
                source = path.read_text(encoding="utf-8")
            self.assertIn("force_eager_attention", source)
            self.assertIn('_attn_implementation = "eager"', source)
            self.assertIn("force_eager_attention(model)", source)

    def test_verify_stdout_is_utf8_safe_on_windows(self):
        source = (ROOT / "scripts" / "verify.py").read_text(encoding="utf-8")
        self.assertIn("sys.stdout.reconfigure", source)
        self.assertIn('encoding="utf-8"', source)


if __name__ == "__main__":
    unittest.main()
