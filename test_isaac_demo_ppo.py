import unittest
from types import SimpleNamespace

import torch

from isaac_demo_ppo import curriculum_settings, install_demo_guided_ppo, checkpoint_demo_state, prepare_resumed_demo_dataset


class DemoPPOTests(unittest.TestCase):
    def test_checkpoint_restores_curriculum_and_correction_dataset(self):
        actor = torch.nn.Linear(36, 8)
        policy = SimpleNamespace(actor=actor)
        alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(actor.parameters()), update=lambda: {})
        env = SimpleNamespace(reset=lambda: None)
        runner = SimpleNamespace(alg=alg, env=SimpleNamespace(unwrapped=env), device="cpu", log_dir=None,
            web_demo_dataset=(torch.ones(4, 36), torch.ones(4, 8)))
        install_demo_guided_ppo(runner, ["demo.npz"], 100,
            {"iteration": 1000, "curriculum_iterations": 1000, "warmup_iterations": 20})
        self.assertAlmostEqual(env.web_demo_variation, 0.05)
        alg.update()
        saved = checkpoint_demo_state(runner, ["demo.npz"], 10.22)
        self.assertEqual(saved["demo_ppo_state"]["iteration"], 1001)
        self.assertEqual(saved["demo_ppo_state"]["curriculum_iterations"], 1000)
        self.assertEqual(saved["demo_ppo_state"]["bc_weight"], 5.0)
        self.assertEqual(saved["demo_paths"], ["demo.npz"])
        self.assertNotIn("losses", saved["demo_ppo_state"])
        restored = SimpleNamespace()
        prepare_resumed_demo_dataset(restored, [], 10.22, saved["demo_dataset"])
        self.assertTrue(torch.equal(restored.web_demo_dataset[0], runner.web_demo_dataset[0]))

    def test_variation_starts_at_zero_and_bc_never_disappears(self):
        self.assertEqual(curriculum_settings(0, 1000), (0.0, 50.0))
        self.assertEqual(curriculum_settings(100, 1000)[0], 0.0)
        self.assertAlmostEqual(curriculum_settings(750, 1000)[0], 0.05)
        self.assertEqual(curriculum_settings(1000, 1000)[1], 5.0)

    def test_warmup_freezes_actor_then_bc_gradient_is_applied(self):
        actor = torch.nn.Linear(36, 8)
        critic = torch.nn.Linear(36, 1)
        policy = torch.nn.Module()
        policy.actor, policy.critic = actor, critic
        policy.log_std = torch.nn.Parameter(torch.full((8,), -3.5))
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
        alg = SimpleNamespace(policy=policy, optimizer=optimizer)
        def update():
            optimizer.zero_grad(set_to_none=True)
            (actor(torch.ones(2, 36)).square().mean() + critic(torch.ones(2, 36)).square().mean()).backward()
            optimizer.step()
            return {"value_function": 1.0}
        alg.update = update
        env = SimpleNamespace(reset=lambda: None)
        runner = SimpleNamespace(alg=alg, env=SimpleNamespace(unwrapped=env), device="cpu", log_dir=None,
            web_demo_dataset=(torch.ones(4, 36), torch.ones(4, 8)))
        install_demo_guided_ppo(runner, [], 10)
        initial = actor.weight.detach().clone()
        alg.update()
        self.assertTrue(torch.equal(initial, actor.weight))
        alg.update()
        self.assertFalse(torch.equal(initial, actor.weight))
        self.assertTrue(torch.equal(policy.log_std.detach(), torch.full((8,), -3.5)))


if __name__ == "__main__":
    unittest.main()
