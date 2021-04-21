import math

__all__ = ['controlvae']


def _Kp_fun(Err, scale=1):
    return 1.0 / (1.0 + float(scale) * math.exp(Err))


class ControlVAE:
    def __init__(self):
        self.I_k1 = 0.0
        self.W_k1 = 0.0
        self.e_k1 = 0.0

    def pid(self, exp_KL, kl_loss, Kp=0.001, Ki=-0.001, Kd=0.01):
        """
        position PID algorithm
        Input: KL_loss
        return: weight for KL loss, beta
        """
        error_k = exp_KL - kl_loss
        # compute U as the control factor
        Pk = Kp * _Kp_fun(error_k)
        Ik = self.I_k1 + Ki * error_k
        # Dk = (error_k - self.e_k1) * Kd

        # window up for integrator
        if 0 > self.W_k1 >= 1:
            Ik = self.I_k1

        Wk = Pk + Ik
        self.W_k1 = Wk
        self.I_k1 = Ik
        self.e_k1 = error_k

        # min and max value
        if Wk > 1:
            Wk = 1.0
        if Wk < 0:
            Wk = 0.0

        return Wk, error_k


def controlvae():
    """
    Constructs a Control VAE model.
    """
    return ControlVAE()


