import yaml


def test_config_has_expected_sections():
    with open("config.yaml", "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    assert "data" in config
    assert "model" in config
    assert "api" in config
    assert config["data"]["n_transactions"] > 0
    assert config["model"]["precision_target"] > 0
