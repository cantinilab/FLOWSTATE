from dataclasses import dataclass
from typing import Dict
import logging

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax._src.random import KeyArray

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import flowstate

from anndata import AnnData
import anndata as ad 
from scipy.stats import ranksums
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from tqdm import tqdm
@dataclass
class DataLoader:
    """DataLoader feeds data from an AnnData object to the model as JAX arrays. It
    samples without replacement for a given batch size.

    Args:
        adata (AnnData): The input AnnData object.
        time_key (str): The obs field with float time observations
        omics_key (str): The obsm field with the omics coordinates.
        batch_size (int): The batch size.
        train_val_split (float, optional): The proportion of train in the split.
        weight_key (str, optional): The obs field with the marginal weights.
    """

    adata: AnnData
    time_key: str
    omics_key: str
    old_omics_key : str
    batch_size: int
    train_val_split: float                                                                    
    exclude_last : bool
    new_loss : bool 
    weight_key: str | None = None
    full_batch : bool = False
   

    def __post_init__(self) -> None:
        """Initialize the DataLoader."""
        print('in the genral loss file')
        # Check that we have a valid time observation.
        assert_msg = "Time observations must be numeric."
        assert self.adata.obs[self.time_key].dtype.kind in "biuf", assert_msg

        # If time is valid, then we can get hold of the timepoints and their indices.
        if self.exclude_last : 
            self.timepoints = np.sort(np.unique(self.adata.obs[self.time_key]))[:-1] # get unique time points 
        else : 
            self.timepoints = np.sort(np.unique(self.adata.obs[self.time_key]))

        print (self.timepoints)
        get_idx = lambda t: np.where(self.adata.obs[self.time_key] == t)[0]
        self.idx = [get_idx(t) for t in self.timepoints] # get the index for each time points 
 
        # Get the number of features, spatial dimensions, and timepoints.
        self.n_features = self.adata.obsm[self.omics_key].shape[1]
        self.n_timepoints = len(self.timepoints)
        nb_cells = sum(len(idx) for idx in self.idx)
        self.size = int(self.batch_size *nb_cells  / self.n_timepoints)
        print("the batch size is :", self.size)
       

    def make_train_val_split(self, key: KeyArray) -> None:
        """Make a train/validation split. Must be called before training.
        self.Idx_train/ self.idx_val = index of train and val 
        Args:
            key (PRNGKey): The random number generator key for permutations.
        """

        # Initialize the train and validation indices, from which we will sample batches.
        self.idx_train, self.idx_val = [], []

        # Iterate over timepoints.
        for idx_t in self.idx:
            # Permute the indices in order to make the split random.
            key, key_permutation = jax.random.split(key)
            permuted_idx = jax.random.permutation(key_permutation, idx_t)

            # Split the indices between train and validation.
            split = int(self.train_val_split * len(idx_t))
            self.idx_train.append(permuted_idx[:split])
            self.idx_val.append(permuted_idx[split:])

        # Log some stats about the split.
        logging.info(f"Train (# cells): {[len(idx) for idx in self.idx_train]}")
        logging.info(f"Val (# cells): {[len(idx) for idx in self.idx_val]}")


    def next(self, key: KeyArray, train_or_val: str) -> Dict[str, jax.Array]:
        """Get the next batch from either train or val indices.

        Args:
            key (KeyArray): The random number generator key for sampling.
            train_or_val (str): Either "train" or "val".

        Returns:
            Dict[str, jax.Array]: A dictionary of JAX arrays.
        """

        # Check that we have a valid train or val argument.
        assert train_or_val in ["train", "val"], "Select either 'train' or 'val'."
        idx = self.idx_train if train_or_val == "train" else self.idx_val

        # Initialize the lists of omics and spatial coordinates over timepoints.
        x,x_old, a = [],[], []

        for idx_t in idx:
            key, key_choice = jax.random.split(key)
            len_t = len(idx_t)


            # if the batch size is smaller or equal to the number of cells n, then we
            # want to sample a minibatch without replacement.
           
            if self.size <= len_t:
                shape = (self.size,)
                batch_idx = jax.random.choice(key_choice, idx_t, shape, replace=False) # pick shape random idx amongst train or val 

                if self.weight_key:  # if weights for diff celll
                    batch_a = self.adata.obs[self.weight_key].iloc[batch_idx].values
                    batch_a /= batch_a.sum()
                else:
                    batch_a = np.ones(shape[0])
                    batch_a /= batch_a.sum()  # Weights are uniform.
                    

            # if the batch size is greater than the number of cells n, then we want
            # to pad the cells with random cells and pad a with zeroes.
            else:
                shape = (self.size - len_t,)
                batch_idx = jax.random.choice(key_choice, idx_t, shape, replace=True)
                batch_idx = np.concatenate((idx_t, batch_idx))

                if self.weight_key:
                    batch_a = self.adata.obs[self.weight_key].iloc[idx_t].values
                    batch_a = np.concatenate((batch_a, np.zeros(shape[0])))
                    batch_a /= batch_a.sum()
                else:
                    batch_a = np.concatenate((np.ones(len_t), np.zeros(shape[0])))
                    batch_a /= batch_a.sum()  # Weights are uniform.
                

            # Get the omics and spatial coordinates for the batch.
            
            
            x.append(self.adata.obsm[self.omics_key][batch_idx]) # embedding
            
            
            if self.new_loss : 
                x_old.append(self.adata.obsm[self.old_omics_key][batch_idx])
            else : 
                x_old.append( [])
            
           
            a.append(batch_a)

        # Return a dictionary of JAX arrays, the first axis being time.
        jnp_stack = lambda x: jnp.array(np.stack(x))
        
       
        return {"x": jnp_stack(x), "a": jnp_stack(a), "x_old" : jnp_stack(x_old)}

    

    def train_or_val(self, iteration: int) -> bool:
        """Sample whether to train or validate.

        Args:
            iteration (int): The current iteration.

        Returns:
            bool: True for train, False for val.
        """
        freq_val = 1 - self.train_val_split
        return iteration % int(1 / freq_val) != 0
    


def compute_potential(
    adata: AnnData,
    model,
    omics_key: str,
    key_added: str = "potential",
) -> None:
    """Compute the potential for all cells in an AnnData object.

    Args:
        adata (AnnData): Input data
        omics_key (str): The omics key
        key_added (str): The obs key to store the potential. Defaults to "potential"

    """
    potential_fn = lambda x: model.potential.apply(model.params, x)
    adata.obs[key_added] = np.array(potential_fn(adata.obsm[omics_key]))


def compute_velocity(
    adata: AnnData,
    model,
    omics_key: str,
    key_added: str = "X_velo",
) -> None:
    """Compute -grad J for all cells in an AnnData object, where J is the potential.

    Args:
        adata (AnnData): Input data
        omics_key (str): The omics key
        key_added (str): The obsm key to store the potential. Defaults to "X_velo"

    """
    potential_fn = lambda x: model.potential.apply(model.params, x)
    velo_fn = lambda x: -jax.vmap(jax.grad(potential_fn))(x)
    adata.obsm[key_added] = np.array(velo_fn(adata.obsm[omics_key]))


def plot_velocity(
    adata: AnnData,
    omics_key: str,
    basis: str,
    velocity_key: str = "X_velo", return_ax=False,
    **kwargs,
) -> None:
    """Plot velocity, as computed by `compute_velocity`
    TBC
    Args:
        adata (AnnData): Input data
        omics_key (str): The obsm key for omics
        velocity_key (str): The obsm key for the velocity
    """
    import cellrank as cr
    vk = cr.kernels.VelocityKernel(
        adata, attr="obsm", xkey=omics_key, vkey=velocity_key
    ).compute_transition_matrix(backend="threading")

    vk.plot_projection(
        basis=basis,
        recompute=True,legend_loc=None,
        **kwargs
    )

    if return_ax:
        return plt.gca()

def select_driver_genes(
    adata, n_stages: int, n_genes: int, regression_key="regression", remove_ones=True
):

    # By default, remove perfect score since they are suspect.
    idx = np.array(adata.var[f"{regression_key}_score"]) != 1.0
    adata_subset = adata[:, idx] if remove_ones else adata

    i_list = np.arange(0, adata_subset.n_obs, adata_subset.n_obs // n_stages)

    gene_names = []
    for k in range(len(i_list) - 1):

        # We'll look for the best genes in this interval ( gene whose regressed expression reaches its max between this interval)
        i_min, i_max = i_list[k], i_list[k + 1]
        order_idx = i_min <= np.array(adata_subset.var[f"{regression_key}_argmax"])
        order_idx &= np.array(adata_subset.var[f"{regression_key}_argmax"]) < i_max

        for j, i in enumerate(
            np.where(order_idx)[0][

                np.argsort(
                    np.array(adata_subset.var[f"{regression_key}_score"])[order_idx]
                )[::-1][:n_genes]
            ]
        ):
            gene_names.append(adata_subset.var_names[i])

    return adata.var.loc[gene_names, f"{regression_key}_argmax"].sort_values().index

def plot_gene_trends(
    adata, gene_names, potential_key="potential", regression_key="regression", title=""
):

    fig, ax = plt.subplots(1, 1)

    X = (
        adata[np.argsort(np.array(adata.obs[potential_key])), gene_names]
        .layers[regression_key]
        .T.copy()
    )

    # Normalize rows
    X = X - X.min(axis=1)[:, None]
    X = X / X.max(axis=1)[:, None]
    implot = ax.imshow(X, aspect="auto", cmap="viridis", interpolation="none")

    # Set gene_names as yticks with small font size
    ax.set_yticks(
        np.arange(0, X.shape[0]),
        gene_names,
        fontsize=6,
    )
    ax.set_xlabel("Cells ordered by potential")

    fig.colorbar(implot)
    plt.title(title)

    return fig, ax

def plot_single_gene_trend(
    adata,
    gene,
    potential_key="potential",
    annotation_key="annotation",
    regression_key="regression",
    show_regression=False,invert_axis=False,
    **kwargs,
):

    sns.scatterplot(
        x=adata.obs[potential_key],
        y=adata[:, gene].X.ravel(),
        hue=adata.obs[annotation_key],
        **kwargs,
    )

    if show_regression:
        xx = adata.obs[potential_key]
        yy = adata[:, gene].layers[regression_key].ravel()
        sns.lineplot(x=xx, y=yy, **kwargs)

    sns.despine()

    plt.title(gene)
    plt.legend(markerscale=3)
    if invert_axis: 
        plt.gca().invert_xaxis()
    plt.show()


def regress_genes(
    adata, potential_key="potential", regression_model=None, key_added="regression"
) -> None:

    # We want to regress gene expression from the potential
    x_train = np.array(adata.obs[potential_key]).reshape(-1, 1).astype(np.float64)

    # The model is a spline regression
    if not regression_model:
        # 5 knots by default 
        regression_model = make_pipeline(
            SplineTransformer(knots="quantile", extrapolation="continue"),
            LinearRegression(),
        )

    adata.layers[key_added] = adata.X.copy()

    # Fit the regression_model for each gene and keep the score and argmax
    for i, gene in tqdm(enumerate(adata.var_names)):

        # The target gene expression
        y_train = adata[:, gene].X.ravel()

        # Fit the regression_model
        regression_model.fit(x_train, y_train)

        # Store the results
        adata.layers[key_added][:, i] = regression_model.predict(x_train)
        adata.var.loc[gene, key_added + "_score"] = regression_model.score(
            x_train, y_train
        )
        # sort the potental in increasing order predict gene exp, gives us a vector of predicted
        # gene expression along the potential (aka sorted by potential). argmax select the index where the predicted gene exp is higher 
        #aka the argmax p mean that gene’s fitted curve reaches its maximum around the p-th smallest potential value.
        adata.var.loc[gene, key_added + "_argmax"] = regression_model.predict(
            np.sort(x_train, axis=0)
        ).argmax()


def tf_enrich(adata, df_tf, regression_key="regression", gene_key=None, n =20):
    
    
    #get only the gene that are TF targets 
    if gene_key:
        df_tf = df_tf[df_tf["target"].isin(adata.var[gene_key].values)]
    else:
        df_tf = df_tf[df_tf["target"].isin(adata.var_names)]

    #build a TF→(is target?) matrix ( 1 if target of TF else 0)
    for tf in tqdm(df_tf["source"].unique()):
        idx = df_tf["source"] == tf
        adata.var[tf] = 0.0

        # Iterate over rows of df_tf[idx]:
        for target in df_tf.loc[idx, "target"]:
            if gene_key:
                adata.var.loc[adata.var[gene_key].values == target, tf] = 1
            else:
                adata.var.loc[adata.var_names == target, tf] = 1

    df_tf_stats = pd.DataFrame(index=df_tf["source"].unique())
    for tf in tqdm(df_tf_stats.index):
        #for each TF get the regresion score for target and non target gene 
        idx_target = adata.var[tf] > 0
        target_scores = adata.var.loc[
            idx_target, f"{regression_key}_score"
        ].values.astype(float)

        idx_nontarget = adata.var[tf] == 0
        nontarget_scores = adata.var.loc[
            idx_nontarget, f"{regression_key}_score"
        ].values.astype(float)
        # check if the target are more corrolated with the potential then the non target 
        stat, p_value = ranksums(target_scores, nontarget_scores, alternative="greater")
        df_tf_stats.loc[tf, ["stat", "p_value", "n_targets"]] = (
            stat,
            p_value,
            idx_target.sum(),
        )

    idx = df_tf_stats["p_value"] < 0.05
    tf_names = df_tf_stats[idx].sort_values("p_value").index[:n]
    sns.barplot(
        y=tf_names.str.upper(), x=-np.log10(df_tf_stats.loc[tf_names, "p_value"])
    )
    plt.ylabel("Transcription factor")
    plt.xlabel(r"$-\log_{10}(p)$")
    plt.title("Transcription factor enrichment scores")
    
    return tf_names.tolist(), df_tf_stats