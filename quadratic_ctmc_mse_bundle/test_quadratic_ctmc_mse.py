import unittest
import numpy as np
from quadratic_ctmc_mse import event_model, pgf, curves, simulate
from limited_ctmc_kbatch_sync_lumped_grouped_metrics import HeterogeneousLumpedKBatchSyncCTMC as CTMC

class Tests(unittest.TestCase):
    def test_marked_self_events_poisson(self):
        model = CTMC([(2, 1.0), (1, .5)], 1, 1, 1)
        events, H = event_model(model)
        t = np.linspace(0, 5, 51)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(H[0, 0], 2.5)
        np.testing.assert_allclose(pgf(model, H, np.ones(1), .8, t), np.exp(-.5*t), atol=1e-12)

    def test_pgf_normalization_and_event_generator(self):
        for K,B,J in [(2,1,1), (3,2,2), (4,2,1)]:
            model = CTMC([(2,1.), (1,.5)], K, B*J, B)
            events, H = event_model(model)
            Q = np.zeros_like(model.Q)
            for i,row in enumerate(events):
                for j,rate,_ in row:
                    Q[i,j] += rate
                    Q[i,i] -= rate
            np.testing.assert_allclose(Q, model.Q)
            pi = model.solve_stationary_distribution()
            np.testing.assert_allclose(pgf(model,H,pi,1.,np.linspace(0,3,11)), 1., atol=1e-12)

    def test_time_mse_poisson(self):
        model = CTMC([(2,1.)],1,1,1)
        events,H = event_model(model)
        times = np.linspace(0,10,31)
        q,floor,dfloor,_,_ = curves(1.,.2,1.,2,1,2.,np.arange(2))
        exact = floor+(4-floor)*pgf(model,H,np.ones(1),q*q,times)
        closed = floor+(4-floor)*np.exp(2*(q*q-1)*times)
        np.testing.assert_allclose(exact,closed,atol=1e-12)
        _,stats = simulate(events,np.ones(1),times,np.arange(2),12000,q,.2,.5,2.,77)
        se = (stats[2]-stats[1])/3.92
        self.assertTrue(np.all(abs(stats[0]-exact) <= 6*se+1e-12))
        bound = dfloor+(4-dfloor)*pgf(model,H,np.ones(1),q,times)
        self.assertTrue(np.all(exact <= bound+1e-12))

    def test_noise_and_round_mse(self):
        rng = np.random.default_rng(9)
        sigma2,m,K = 2.,4,3
        noises = rng.normal(0,np.sqrt(sigma2/m),size=(150000,K)).mean(axis=1)
        self.assertLess(abs(noises.var()/(sigma2/(m*K))-1), .015)
        model = CTMC([(1,1.)],1,1,1)
        events,_ = event_model(model)
        rounds = np.arange(51)
        q,floor,_,exact,bound = curves(1.,.1,sigma2,m,K,2.,rounds)
        stats,_ = simulate(events,np.ones(1),np.array([0.,.1]),rounds,15000,q,.1,sigma2/(m*K),2.,123)
        se = (stats[2]-stats[1])/3.92
        self.assertTrue(np.all(abs(stats[0]-exact) <= 6*se+1e-12))
        self.assertTrue(np.all(exact <= bound+1e-12))

if __name__ == '__main__':
    unittest.main()
