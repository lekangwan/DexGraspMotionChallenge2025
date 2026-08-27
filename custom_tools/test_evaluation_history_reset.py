from evaluation_loop import reset_policy_inference_state


class MockPolicy:
    def __init__(self):
        self.reset_count = 0

    def reset_inference_history(self):
        self.reset_count += 1


class MockLightningModel:
    def __init__(self):
        self.model = MockPolicy()


def test_reset_nested_policy_history():
    model = MockLightningModel()
    reset_policy_inference_state(model)
    assert model.model.reset_count == 1


def test_policy_without_history_is_accepted():
    reset_policy_inference_state(object())
