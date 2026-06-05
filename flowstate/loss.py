from .steps.proximal_step import ProximalStep
import flax.linen as nn
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.linear.implicit_differentiation import ImplicitDiff
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
from ott.geometry.pointcloud import PointCloud
from typing import Dict
from ott.problems.linear.linear_problem import LinearProblem
import jax.numpy as jnp
import optax
import jax
from jax import lax
import numpy as np

def linear_loss(
    x: jax.Array,
    a: jax.Array,
    y: jax.Array,
    b: jax.Array,
    epsilon: float,
    debias: bool,
) -> float:
    """Compute the Sinkhorn loss (no quadratic component).

    Args:
        x: A pointcloud.
        a: Histogram on x.
        y: Another pointcloud.
        b: Histogram on y.
        epsilon: Entropic regularization parameter.
        debias: Whether to debias the loss.

    Returns:
        The Sinkhorn loss.
    """

    # Define geometries, compute epsilon relative to the yy geometry.
    # For Sinkhorn, epsilon is defined in the Geometry.
    # For FGW, it is defined in the solver.
    geom_yy = PointCloud(y, y, epsilon=epsilon)
    geom_xx = PointCloud(x, x).copy_epsilon(geom_yy)
    geom_xy = PointCloud(x, y).copy_epsilon(geom_yy)

    # Define some hyperparameters.
    implicit_diff = ImplicitDiff(symmetric=True)

    # Compute the Sinkhorn loss between point clouds x and y.
    problem = LinearProblem(geom_xy, a=a, b=b)
    ott_solver = Sinkhorn(implicit_diff=implicit_diff)
    ot_loss = ott_solver(problem).reg_ot_cost

    # We assume x and y to have the same mass, so no need for the m(a) - m(b) term.
    if debias:
        # Debias the Sinkhorn loss with the xx term.
        problem = LinearProblem(geom_xx, a=a, b=a)
        ott_solver = Sinkhorn(implicit_diff=implicit_diff)
        ot_loss -= 0.5 * ott_solver(problem).reg_ot_cost

        # Debias the Sinkhorn loss with the yy term.
        problem = LinearProblem(geom_yy, a=b, b=b)
        ott_solver = Sinkhorn(implicit_diff=implicit_diff)
        ot_loss -= 0.5 * ott_solver(problem).reg_ot_cost

    return ot_loss


def loss_fn(
    params: optax.Params,
    batch: Dict[str, jax.Array],
    teacher_forcing: bool,
    proximal_step: ProximalStep,
    potential: nn.Module,
    n_steps: int,
    epsilon: float,
    debias: bool,
    new_loss : bool,
    l: int = 1,
    time_old : jax.Array= None, 
    velo : str = "accumulated",
) -> jax.Array:
    """The loss function

    Args:
        params: The parameters of the model.
        batch: A batch of data.
        teacher_forcing: Whether to use teacher forcing.
        proximal_step: The proximal step, e.g. LinearExplicitStep.
        potential: The potential function parametrized by a neural network.
        n_steps: The number of steps to take.
        epsilon: Entropic regularization parameter.
        debias: Whether to debias the loss (see linear_loss or quadratic_loss).
        fused_penalty: Parameter indicting weight of the fused term.

    Returns:
        The loss and the data for the next iteration ( pred if no teacher forcing and grounfd truth if teacher forcing) 
    """

    # This is a helper function to compute the loss for a single timepoint.
    # We will chain this function over the timepoints using lax.scan.
    def _through_time(carry, t):
        # Unpack the carry, which contains the x and space across timepoints.
        _x, _a, _x_old = carry # a is histogramme 
        #jax.debug.print("Timepoint: {}", t)
        # Predict the timepoint t+1 using the proximal step.
        
        pred_x = proximal_step.chained_training_steps(_x[t], _a[t], potential, params, n_steps)
        #pred_x = proximal_step.chained_training_steps(_x[t], _a[t], potential, params, 1, n_steps)
        
        ot_loss = linear_loss(
                x=pred_x,
                a=_a[t],
                y=_x[t + 1],
                b=_a[t + 1],
                epsilon=epsilon,
                debias=debias,
            )
        if new_loss: 

            if velo == "accumulated":
                _x_old_pred = _x_old[t + 1]
                num_steps = t+1 - time_old[t+1]  
                max_steps =len(_x ) 

                def scan_body(carry, i):
                    
                    do_step = i <num_steps  # traced boolean

                    def do_update(_):
                        return proximal_step.chained_training_steps(carry, _a[t + 1], potential, params,  n_steps)

                    def skip_update(_):
                        return carry  # just pass through

                    carry = lax.cond(do_step, do_update, skip_update, operand=None)
                    return carry, None
                _x_old_pred, _ = lax.scan(scan_body, _x_old_pred, jnp.arange(max_steps))
                
                #first case
                n = len(_x[t + 1])
                ot_loss = (ot_loss + (l / n) * jnp.linalg.norm(_x[t + 1] - _x_old_pred)**2) / (1 + l)
            if velo == "instantaneous":
                # second case
                n = len(_x[t ])
                u = pred_x - _x[t] # is the potential 
                v = _x_old[t] # is the velocity change name 
                ot_loss = (ot_loss + (l / n) * jnp.linalg.norm((v - u))**2 / (1 + l))


        # If no teacher-forcing, replace next observation with predicted
        replace_fn = lambda u: u.at[t + 1].set(pred_x)
        _x = jax.lax.cond(teacher_forcing, lambda u: u, replace_fn, _x)

        # And the same thing for the histogram.
        replace_fn = lambda u: u.at[t + 1].set(_a[t])
        _a = jax.lax.cond(teacher_forcing, lambda u: u, replace_fn, _a)

        # Return the data for the next iteration and the current loss.
        return (_x, _a, _x_old), ot_loss

    # Iterate through timepoints efficiently. ot_loss becomes a 1-D array.
    # Notice that we do not compute the loss for the last timepoint, because there
    # is no next observation to compare to.
    
    timepoints = jnp.arange(len(batch["x"]) - 1)
    
    init_carry = (batch["x"], batch["a"], batch["x_old"], )
    _, ot_loss = jax.lax.scan(_through_time, init_carry, timepoints)

    # Sum the losses over all timepoints, 
    return jnp.sum( ot_loss)
