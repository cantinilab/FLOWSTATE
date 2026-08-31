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

from typing import Sequence
import plotly.graph_objects as go
import plotly.express as px
from anndata import AnnData
import anndata as ad 
from scipy.stats import ranksums
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from tqdm import tqdm

from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.linear.implicit_differentiation import ImplicitDiff
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem

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
    velo_omics_key : str
    batch_size: int
    train_val_split: float                                                                    
    exclude_last : bool
    new_loss : bool 
    weight_key: str | None = None
    
   

    def __post_init__(self) -> None:
        """Initialize the DataLoader."""
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
                    batch_a = self.adata.obs[self.weight_key].iloc[batch_idx].values.copy()
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
                x_old.append(self.adata.obsm[self.velo_omics_key][batch_idx])
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


def pred(
    adata: ad.AnnData,
    time_key: str,
    model,
    omics_key: str
):
    """Transform the data given a subet of batches."""

    # List timepoints and time differences between them.
    timepoints = np.sort(adata.obs[time_key].unique())
    

    # Iterate over timepoints and transform the data.
    for i, t in enumerate(timepoints[:-1]):
        idx = (adata.obs[time_key] == t) 
        adata.obsm["pred"][idx] = model.transform(
            adata[idx], omics_key=omics_key, batch_size=1000
        )

def OT_plan(adata, time_obs, omics_key, epsilon=0.05,  weight_key = None ):
    
    # List timepoints and time differences between them.
    timepoints = np.sort(adata.obs[time_obs].unique())
    t_diff = np.diff(timepoints).astype(float)
    #initialize Ot_plan 
    P = sp.lil_matrix((adata.n_obs, adata.n_obs), dtype=float)

    # For Sinkhorn, epsilon is defined in the Geometry.
    # For FGW, it is defined in the solver.
    
    
    for i, t in enumerate(timepoints[:-1]):
        idx = (adata.obs[time_obs] == t) 
        idx_next = (adata.obs[time_obs] == timepoints[i + 1])

        if weight_key:  # if weights for diff celll
            a = adata.obs.loc[idx, weight_key].values
            a /= a.sum()

            b = adata.obs.loc[idx_next, weight_key].values
            b /= b.sum()
            
        else:
            a = np.ones(idx.sum()) / idx.sum()
            b = np.ones(idx_next.sum()) / idx_next.sum()

        x= adata[idx].obsm["pred"]
        y=  adata[idx_next].obsm[omics_key]

        geom_yy = PointCloud(y, y, epsilon=epsilon)
        geom_xx = PointCloud(x, x).copy_epsilon(geom_yy)
        geom_xy = PointCloud(x, y).copy_epsilon(geom_yy)

        # Define some hyperparameters.
        implicit_diff = ImplicitDiff(symmetric=True)

        # Compute the Sinkhorn loss between point clouds x and y.
        problem = LinearProblem(geom_xy, a=a, b=b)
        ott_solver = Sinkhorn(implicit_diff=implicit_diff)
        solver = ott_solver(problem)
        plan = solver.matrix

        rows = np.where(idx)[0]
        cols = np.where(idx_next)[0]
        P[np.ix_(rows, cols)] = plan

    adata.obsp["OT_plan"] = P.tocsr()
     # (optional) keep a small mapping to retrieve which rows/cols correspond to each timepoint
    adata.uns.setdefault("OT_plan_meta", {})
    adata.uns["OT_plan_meta"]["timepoints"] = timepoints.tolist()
    return adata

def predict_transition_probabilities_KNN(
    adata,
    time_key: str,
    type_key: str,
    pred_key: str = "pred",
    omics_key: str = "X",
    k: int = 10,
):
    """
    Train one global kNN classifier on all real cells.
    Then, for each timepoint t, predict cell-type probabilities for
    predicted cells at t -> t+Δt.
    Returns a DataFrame of transition probabilities (cell × target cell type).
    """
    
    timepoints = np.sort(adata.obs[time_key].unique())
    all_results = []

    # ------------------------------------------------------------
    # 1) TRAIN kNN ON THE ENTIRE REAL DATASET
    # ------------------------------------------------------------
    X_real_all = adata.obsm[omics_key]
    y_real_all = adata.obs[type_key].values

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_real_all, y_real_all)

    classes = knn.classes_

    # ------------------------------------------------------------
    # 2) PREDICT PROBABILITIES PER TIMEPOINT
    # ------------------------------------------------------------
    for i, t in enumerate(timepoints[:-1]):
        next_t = timepoints[i + 1]

        # Extract predicted data (at time t, evolved to next_t)
        idx_pred = (adata.obs[time_key] == t)
        X_pred = adata.obsm[pred_key][idx_pred]

        # Predict probabilities for each predicted cell
        probs = knn.predict_proba(X_pred)

        # Build results DataFrame
        prob_df = pd.DataFrame(
            probs,
            index=adata.obs_names[idx_pred],
            columns=classes
        )

        # Add metadata
        prob_df["time_t"] = t
        prob_df["time_t+dt"] = next_t
        prob_df["source_type"] = adata.obs.loc[idx_pred, type_key].values

        # Pick the most likely transition (argmax over columns)
        prob_cols = prob_df.columns.difference(["time_t", "time_t+dt", "source_type"])
        prob_df["pred_target_type"] = prob_df[prob_cols].idxmax(axis=1)

        all_results.append(prob_df)

    return pd.concat(all_results)

def plot_transition_flow_chart(adata, transition_probs, weight_key=None,
    time_key=None,
    color_map=None,
    top_n=None,normalization_type="total", min_flow=0.005):
    """
    Create a Sankey flow chart of weighted cell-type transitions across consecutive timepoints.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing per-cell weights in `adata.obs[weight_key]`
        and time labels in `adata.obs[time_key]`.
    transition_probs : pd.DataFrame
        DataFrame indexed by cell IDs (matching `adata.obs_names`) and containing:
        - 'time_t'
        - 'time_t+dt'
        - 'source_type'
        - 'pred_target_type'
    weight_key : str
        Column in `adata.obs` with per-cell weights.
    time_key : str
        Column in `adata.obs` with timepoint labels.
    color_map : dict
        Mapping {cell_type: color} used to color nodes (e.g. hex strings).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly Sankey figure (also displayed if in a notebook).
    """

    if time_key is None:
        raise ValueError("time_key must be provided (column in adata.obs).")

    # Collect all flows between consecutive timepoints
    flows = []

    timepoints = np.sort(adata.obs[time_key].unique())
    for i, t in enumerate(timepoints[:-1]):
        next_t = timepoints[i + 1]

        # Subset transition probabilities for this time transition
        subset = transition_probs[
            (transition_probs["time_t"] == t) &
            (transition_probs["time_t+dt"] == next_t)
        ].copy()

        # weights (weighted if key provided, else unweighted)
        if (weight_key is None) :
            weights = pd.Series(1.0, index=subset.index)
        else:
            weights = adata.obs.loc[subset.index, weight_key]

        if normalization_type == "source":
            # Weighted contingency table
            weighted_table = (
                pd.DataFrame({
                    "source_type": subset["source_type"],
                    "pred_target_type": subset["pred_target_type"],
                    "weight": weights,
                })
                .pivot_table(
                    index="source_type",
                    columns="pred_target_type",
                    values="weight",
                    aggfunc="sum",
                    fill_value=0,
                )
            )

            # Normalize rows by total weight to get fractions
            transition_counts = weighted_table.div(weighted_table.sum(axis=1), axis=0)

        elif  normalization_type == "total":
            #Build weighted long-format df
            weighted_df = pd.DataFrame({
                "source_type": subset["source_type"].values,
                "pred_target_type": subset["pred_target_type"].values,
                "weight": weights.values,
            })

            # Weighted transition table
            weighted_transition = (
                weighted_df.pivot_table(
                    index="source_type",
                    columns="pred_target_type",
                    values="weight",
                    aggfunc="sum",
                    fill_value=0
                )
            )

            # Total weighted mass per source type
            total_source_weight = weighted_df.groupby("source_type")["weight"].sum()
            total_weight = total_source_weight.sum()

            # Fraction of whole system contributed by each source (weighted)
            source_fraction = total_source_weight / total_weight

            # Multiply each row by its global weighted fraction
            for src in weighted_transition.index:
                weighted_transition.loc[src, :] *= source_fraction[src]

            # Normalize whole matrix so sum = 1
            transition_counts = weighted_transition / weighted_transition.values.sum()

        else :
            raise ValueError(f"Unknown normalization_type: {normalization_type}")

        # Store nonzero flows
        for source in transition_counts.index:
            row = transition_counts.loc[source]
            if top_n is not None:
               row= row.sort_values(ascending=False).head(top_n)
            for target,value in row.items():
                
                if value > 0:
                    flows.append((f"{source} (t={t})", f"{target} (t={next_t})", float(value)))

    if len(flows) == 0:
        raise ValueError("No nonzero flows found. Check your inputs (timepoints, transition_probs, weights).")
    
    # Optional: remove tiny flows
    if min_flow is not None:
        flows = [(s, t, v) for (s, t, v) in flows if v > min_flow]
        
    # Node labels and indices
    labels = list(pd.unique([x for flow in flows for x in flow[:2]]))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    source_idx = [label_to_idx[s] for s, _, _ in flows]
    target_idx = [label_to_idx[t] for _, t, _ in flows]
    values = [v for _, _, v in flows]

        # Generate a color map automatically if none is provided
    if color_map is None:
        cell_types = pd.unique(
            [x.split(" (t=")[0] for flow in flows for x in flow[:2]]
        )
        palette = px.colors.qualitative.Plotly
        color_map = {
            cell_type: palette[i % len(palette)]
            for i, cell_type in enumerate(cell_types)
        }

    node_colors = [color_map[label.split(" (t=")[0]] for label in labels]
    link_colors = [color_map[s.split(" (t=")[0]] for s, _, _ in flows]

    node_colors = [color_map[label.split(" (t=")[0]] for label in labels]
    link_colors = [color_map[s.split(" (t=")[0]] for s, _, _ in flows]

    # Plot Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=source_idx,
            target=target_idx,
            value=values,
            color=link_colors,
        )
    )])

    fig.update_layout(title_text="Cell Type Transitions Across Timepoints", font_size=12, width=1500,
    height=700)
    fig.show()
    return fig


