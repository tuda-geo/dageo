import torch

"""
Randomized Maximum Likelihood (RML) method implementation using PyTorch.

This module provides an implementation of the RML method which addresses
inverse problems via
gradient-based optimization. The method solves for m that minimizes:

    J(m) = 0.5 * || forward(m) - (data_obs + noise_d) ||^2_R +
           0.5 * || m - (model_prior + noise_m) ||^2_P

where noise_d ~ N(0, sigma^2) and noise_m ~ N(0, prior_std^2).
Optionally, an external gradient computation can be provided via the
compute_gradient argument.
"""


def default_rml_objective(m, forward, data_pert, sigma, m_prior, prior_std):
    """
    Computes the default RML objective function.

    Args:
       m: model parameter (torch tensor)
       forward: forward operator, callable that takes m and returns
               prediction as torch tensor.
       data_pert: perturbed observed data (torch tensor)
       sigma: observation noise standard deviation (torch tensor or float)
       m_prior: perturbed prior model (torch tensor)
       prior_std: prior noise standard deviation (float or torch tensor)

    Returns:
       loss: objective value (torch tensor)
    """
    pred = forward(m)
    # Data misfit term
    data_term = 0.5 * torch.sum(((pred - data_pert) / sigma) ** 2)
    # Prior regularization term
    prior_term = 0.5 * torch.sum(((m - m_prior) / prior_std) ** 2)
    loss = data_term + prior_term
    return loss


class RMLSolver:
    """
    Class for solving the Randomized Maximum Likelihood (RML) problem using
    gradient-based optimization.
    """

    def __init__(
        self,
        forward,
        data_obs,
        sigma,
        model_prior,
        n_iter=1000,
        lr=1e-2,
        prior_std=1.0,
        optimizer_class=torch.optim.Adam,
        compute_gradient=None,
        verbose=True,
        callback=None,
        device="cpu",
        random_seed=None,
    ):
        """
        Initializes the RMLSolver.

        Args:
            forward: callable, forward operator.
            data_obs: observed data (array-like or torch tensor).
            sigma: observation noise standard deviation (float or array-like).
            model_prior: prior model (array-like or torch tensor).
            n_iter: number of iterations.
            lr: learning rate.
            prior_std: standard deviation for the prior noise.
            optimizer_class: torch optimizer class to use.
            compute_gradient: function to compute gradient externally.
                             Should have signature:
                             compute_gradient(m, forward, data_pert,
                                            sigma, m_prior, prior_std)
            verbose: if True, prints optimization progress.
            callback: optional callback function with signature
                     callback(m, loss, iteration).
            device: 'cpu' or 'cuda'.
            random_seed: optional random seed (int).
        """
        self.forward = forward
        self.n_iter = n_iter
        self.lr = lr
        self.prior_std = prior_std
        self.optimizer_class = optimizer_class
        self.compute_gradient = compute_gradient
        self.verbose = verbose
        self.callback = callback
        self.device = device

        if random_seed is not None:
            torch.manual_seed(random_seed)

        # Convert inputs to torch tensors on the specified device
        if not torch.is_tensor(data_obs):
            self.data_obs = torch.tensor(
                data_obs, dtype=torch.float32, device=device
            )
        else:
            self.data_obs = data_obs.to(device, dtype=torch.float32)

        if not torch.is_tensor(sigma):
            self.sigma = torch.tensor(
                sigma, dtype=torch.float32, device=device
            )
        else:
            self.sigma = sigma.to(device, dtype=torch.float32)

        if not torch.is_tensor(model_prior):
            self.model_prior = torch.tensor(
                model_prior, dtype=torch.float32, device=device
            )
        else:
            self.model_prior = model_prior.to(device, dtype=torch.float32)

        # Perturb observed data and prior model
        self.data_pert = self.data_obs + self.sigma * torch.randn_like(
            self.data_obs
        )
        self.m_prior_pert = (
            self.model_prior
            + self.prior_std * torch.randn_like(self.model_prior)
        )

        # Initialize model parameter starting from perturbed prior
        self.m = self.m_prior_pert.clone().detach().requires_grad_(True)

        self.optimizer = self.optimizer_class([self.m], lr=self.lr)

    def solve(self):
        """
        Runs the optimization to solve the RML problem.

        Returns:
            m_est: the optimized model (torch tensor).
            objective_history: list of loss values over iterations.
        """
        objective_history = []
        for i in range(self.n_iter):
            self.optimizer.zero_grad()
            loss = default_rml_objective(
                self.m,
                self.forward,
                self.data_pert,
                self.sigma,
                self.m_prior_pert,
                self.prior_std,
            )
            if self.compute_gradient is None:
                loss.backward()
            else:
                # Use external gradient computation
                grad = self.compute_gradient(
                    self.m,
                    self.forward,
                    self.data_pert,
                    self.sigma,
                    self.m_prior_pert,
                    self.prior_std,
                )
                self.m.grad = grad
            self.optimizer.step()
            objective_history.append(loss.item())
            if self.verbose and (i % max(1, self.n_iter // 10) == 0):
                print(
                    f"Iteration {i + 1}/{self.n_iter}, Loss: {loss.item():.6f}"
                )
            if self.callback is not None:
                self.callback(self.m, loss, i)
        return self.m.detach(), objective_history


def rml(
    forward,
    data_obs,
    sigma,
    model_prior,
    n_iter=1000,
    lr=1e-2,
    prior_std=1.0,
    optimizer_class=torch.optim.Adam,
    compute_gradient=None,
    verbose=True,
    callback=None,
    device="cpu",
    random_seed=None,
):
    """
    Randomized Maximum Likelihood (RML) method using gradient-based
    optimization.

    This function solves the inverse problem:
        m_est = argmin_m { 0.5 * || forward(m) - (data_obs + noise) ||^2_R +
                           0.5 * || m - (model_prior + noise) ||^2_P }
    where noise terms are drawn from N(0, sigma^2) and N(0, prior_std^2)
    respectively.

    Args:
       forward: callable, forward model that takes a torch tensor and
               returns a torch tensor.
       data_obs: observed data (array-like or torch tensor).
       sigma: standard deviation for observation noise (float or array-like).
       model_prior: prior model (array-like or torch tensor).
       n_iter: number of optimization iterations.
       lr: learning rate for the optimizer.
       prior_std: standard deviation for prior noise.
       optimizer_class: torch optimizer class to use
                       (default: torch.optim.Adam).
       compute_gradient: optional function to compute gradient externally.
       verbose: if True, prints progress.
       callback: optional callback with signature callback(m, loss, iteration).
       device: device to run the optimization on ('cpu' or 'cuda').
       random_seed: optional random seed for reproducibility.

    Returns:
       m_est: the optimized model as a torch tensor.
       objective_history: list of objective loss values over iterations.
    """
    solver = RMLSolver(
        forward=forward,
        data_obs=data_obs,
        sigma=sigma,
        model_prior=model_prior,
        n_iter=n_iter,
        lr=lr,
        prior_std=prior_std,
        optimizer_class=optimizer_class,
        compute_gradient=compute_gradient,
        verbose=verbose,
        callback=callback,
        device=device,
        random_seed=random_seed,
    )
    return solver.solve()
