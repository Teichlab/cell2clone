import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests


# Sum expression across a gene set for each cell (dense or sparse-safe).

def _get_sum_vector(adata, genes):
    """
    Sum expression across a gene set for each cell.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression values.
    genes : list of str
        Gene names to sum.

    Returns
    -------
    numpy.ndarray
        Per-cell summed expression values.
    """
    
    # Sum expression across a gene set for each cell (dense or sparse-safe)
    genes = [g for g in genes if g in adata.var_names]
    if len(genes) == 0:
        raise ValueError("No requested genes found in adata.var_names")

    idx = adata.var_names.get_indexer(genes)
    X = adata[:, idx].X

    if hasattr(X, "tocsr"):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X.sum(axis=1)).ravel()


# Classify cells into TRBC1/TRBC2 or kappa/lambda states
# and compute per-cell log-ratio metrics.

def classify_lightchain_or_trbc(
    adata,
    mode="T",                 # "T" or "B"
    out_prefix=None,          # optional; defaults to "TRBC" or "KL"
):
    """
    Classify cells by receptor-chain usage.

    For T cells, assigns TRBC1 or TRBC2. For B cells, assigns kappa or
    lambda based on summed expression. Stores per-cell classifications
    and log-ratio metrics in ``adata.obs``.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    mode : {"T", "B"}, default="T"
        Whether to classify T-cell receptor beta chains or B-cell light
        chains.
    out_prefix : str, optional
        Prefix for output columns.

    Returns
    -------
    None
    """
    
    # Classify cells into receptor-chain states using hard expression calls
    # and write per-cell metrics to adata.obs.
    
    mode = mode.upper()
    if mode not in {"T", "B"}:
        raise ValueError("mode must be 'T' or 'B'")

    if mode == "T":
        genes_1 = ["TRBC1"]
        genes_2 = ["TRBC2"]
        label1, label2 = "TRBC1", "TRBC2"
        if out_prefix is None:
            out_prefix = "TRBC"

    if mode == "B":
        genes_1 = [g for g in adata.var_names if g.startswith("IGKC")]
        genes_2 = [g for g in adata.var_names if g.startswith("IGLC")]
        label1, label2 = "kappa", "lambda"
        if out_prefix is None:
            out_prefix = "KL"

    s1 = _get_sum_vector(adata, genes_1)
    s2 = _get_sum_vector(adata, genes_2)

    # Store chain-specific counts with intuitive names
    if mode == "B":
        adata.obs["kappa_sum"] = s1
        adata.obs["lambda_sum"] = s2
    
    elif mode == "T":
        adata.obs["TRBC1"] = s1
        adata.obs["TRBC2"] = s2

    cls = np.full(adata.n_obs, "ambiguous", dtype=object)
    
    cls[s1 > s2] = label1
    cls[s2 > s1] = label2
    
    adata.obs[f"{out_prefix}_class"] = cls
    
    # log ratio
    eps = 1e-6
    log_ratio = np.log((s1 + eps) / (s2 + eps))
    adata.obs[f"{out_prefix}_log_ratio"] = log_ratio

    return None  # no model


# Compute sample × celltype clonality metrics from
# classified receptor-chain states (e.g. TRBC1/TRBC2 or kappa/lambda).

def compute_sample_celltype_clonality(
    adata,
    class_col,
    sample_col="sample_id",
    celltype_col="cell_type",
    state_labels=("TRBC1", "TRBC2"),
    min_cells=20,
    out_prefix=None,
    eps=1e-6,
    # --- clone-size score options ---
    clone_score_method="per_celltype",   # "per_celltype" or "global"
    clone_score_col="clone_size_score",
    clone_score_clip_eps=1e-6,
    add_clone_score_center=True,
):
    """
    Compute sample-by-cell-type clonality metrics from two-state receptor
    classifications.

    Calculates state fractions, entropy, log-ratio, clonality, and a
    clone-size score for each sample-cell type combination.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing cell annotations.
    class_col : str
        Column containing receptor-chain classifications.
    ...

    Returns
    -------
    pandas.DataFrame
        Sample × cell type clonality metrics.
    
    Adds:
      - {out_prefix}_log2_ratio = log2((state1 + eps) / (state2 + eps))
      - {out_prefix}_log2_ratio_scaled = median-centered log2 ratio
      - clonality = abs({out_prefix}_log2_ratio_scaled)
      - frac_{state1} (e.g. frac_TRBC1)
      - clone_size_score (bias-centered distance of frac_{state1} to median, scaled to [0,1])
    """
    if out_prefix is None:
        out_prefix = class_col

    state1, state2 = state_labels

    # total cells per sample (ALL cells)
    sample_totals_all = adata.obs.groupby(sample_col, observed=True).size()
    sample_totals_all.name = "n_cells_sample_all"

    # keep only confidently classified cells for state counts
    obs = adata.obs[adata.obs[class_col].isin(state_labels)].copy()

    # sample × celltype × state counts
    df = (
        obs
        .groupby([sample_col, celltype_col, class_col], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for s in state_labels:
        if s not in df.columns:
            df[s] = 0

    df["n_cells"] = df[state1] + df[state2]

    # attach ALL-cells sample totals
    df = df.merge(sample_totals_all.reset_index(), on=sample_col, how="left")

    # fraction of sample cells in this celltype (denominator = ALL cells in sample)
    df["frac_cells_in_sample"] = df["n_cells"] / df["n_cells_sample_all"]

    # filter low-count groups (based on classified cells used for clonality)
    df = df[df["n_cells"] >= min_cells].copy()

    # fractions + entropy metrics (based on classified counts only)
    p = df[state1] / df["n_cells"]
    p = p.clip(clone_score_clip_eps, 1 - clone_score_clip_eps)

    frac_state1_col = f"frac_{state1}"
    df[frac_state1_col] = p
    df[f"{out_prefix}_abs_skew"] = (p - 0.5).abs()
    df[f"{out_prefix}_entropy"] = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    # raw log2 ratio
    log2_col = f"{out_prefix}_log2_ratio"
    df[log2_col] = np.log2((df[state1] + eps) / (df[state2] + eps))

    # median-centered (scaled) log2 ratio
    scaled_col = f"{out_prefix}_log2_ratio_scaled"
    df[scaled_col] = df[log2_col] - df[log2_col].median()

    # directionless clonality
    df["clonality"] = df[scaled_col].abs()

    # ---- clone size score from frac_{state1} ----
    p_col = frac_state1_col
    p2 = df[p_col].astype(float).clip(clone_score_clip_eps, 1 - clone_score_clip_eps)

    if clone_score_method == "per_celltype":
        m = df.groupby(celltype_col, observed=True)[p_col].transform("median").astype(float)
        m = m.clip(clone_score_clip_eps, 1 - clone_score_clip_eps)
    elif clone_score_method == "global":
        m0 = float(np.nanmedian(df[p_col].values))
        m = np.full(len(df), m0, dtype=float)
        m = np.clip(m, clone_score_clip_eps, 1 - clone_score_clip_eps)
    else:
        raise ValueError("clone_score_method must be 'per_celltype' or 'global'")

    dist = (p2 - m).abs()
    max_dist = np.maximum(m, 1 - m)
    df[clone_score_col] = (dist / max_dist).clip(0, 1)

    if add_clone_score_center:
        df[clone_score_col + "_center"] = m

    return df


# Test whether cell types exceed a reference cell type
# for a given metric using label permutation.

def permutation_test(
    df,
    reference_celltype,
    value_col="clone_size_score",
    celltype_col="cell_type",
    sample_col="sample_id",
    percentile=0.99,
    n_perms=10_000,
    min_samples=10,
    random_state=0,
):
    """
    Test whether a cell type contains more highly clonal samples than a
    reference cell type.

    The threshold is defined from a percentile of the reference cell
    type, and significance is assessed by permutation of cell-type
    labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Sample-level clonality metrics.
    reference_celltype : str
        Cell type used to define the threshold.
    ...

    Returns
    -------
    pandas.DataFrame
        Permutation test results with raw and FDR-adjusted p-values.
    """
    rng = np.random.default_rng(random_state)

    # reference threshold
    ref_vals = df.loc[
        df[celltype_col] == reference_celltype,
        value_col
    ].dropna()

    if ref_vals.empty:
        raise ValueError(f"No data for reference celltype {reference_celltype}")

    threshold = np.quantile(ref_vals, percentile)

    data = df[[sample_col, celltype_col, value_col]].dropna().copy()

    results = []

    celltypes = data[celltype_col].unique()

    for ct in celltypes:
        if ct == reference_celltype:
            continue

        sub = data[data[celltype_col] == ct]
        n_samples = sub[sample_col].nunique()
        if n_samples < min_samples:
            continue

        obs_frac = np.mean(sub[value_col] > threshold)
        obs_n = int((sub[value_col] > threshold).sum())

        # permutation null: shuffle celltype labels
        perm_fracs = np.empty(n_perms)
        for i in range(n_perms):
            perm_ct = rng.permutation(data[celltype_col].values)
            perm_sub = data[value_col].values[perm_ct == ct]
            perm_fracs[i] = np.mean(perm_sub > threshold)

        pval = (np.sum(perm_fracs >= obs_frac) + 1) / (n_perms + 1)

        results.append({
            "celltype": ct,
            "reference_celltype": reference_celltype,
            "percentile": percentile,
            "threshold": threshold,
            "n_samples": n_samples,
            "n_exceeding": obs_n,
            "pct_exceeding": obs_frac * 100,
            "p_perm": pval,
        })

        res = pd.DataFrame(results)

        if not res.empty:
            res["q_perm"] = multipletests(
                res["p_perm"].values,
                method="fdr_bh"
            )[1]

    return (
        res
        .sort_values("p_perm")
        .reset_index(drop=True)
    )


# Plot receptor-chain skew using cell-type colours stored in adata.uns.

def plot_chain_skew_violin(
    df,
    adata,
    celltype_col="cell_type",
    mode="B",
    metric=None,
    left_col=None,
    right_col=None,
    clonality_col="clonality",
    frac_col="frac_cells_in_sample",
    order=None,
    order_by=None,
    xlim=None,
    eps=1e-6,
    size_min=2,
    size_max=20,
    jitter=0.3,
    alpha=0.7,
    random_state=0,
    figsize=None,
    show=True,
):
    """
    Plot sample-level receptor-chain skew across cell types.

    Displays violin plots with overlaid sample points, where point size
    represents the fraction of cells contributed by each cell type.

    Parameters
    ----------
    df : pandas.DataFrame
        Sample-level receptor-chain metrics.
    adata : AnnData
        AnnData object providing cell-type colours.
    ...

    Returns
    -------
    matplotlib.figure.Figure
    matplotlib.axes.Axes
    """
    df = df.copy()
    mode = mode.upper()

    # get colours from adata
    categories = adata.obs[celltype_col].cat.categories
    colors = adata.uns[f"{celltype_col}_colors"]
    color_map = dict(zip(categories, colors))

    if mode == "B":
        metric = metric or "KL_class_log2_ratio_scaled"
        left_col = left_col or "kappa"
        right_col = right_col or "lambda"
        xlabel = "log2(kappa / lambda) from classified-cell counts (sample × celltype)"
        title = "Sample-level kappa/lambda skew by B-cell subtype\n(dot size = fraction of sample B cells)"
        # xlim = (-31, 31) if xlim is None else xlim
        order_by = order_by or "median_clonality"

    elif mode == "T":
        metric = metric or "TRBC_class_log2_ratio_scaled"
        left_col = left_col or "TRBC1"
        right_col = right_col or "TRBC2"
        xlabel = "log2(TRBC1 / TRBC2) from classified-cell counts (sample × celltype)"
        title = "Sample-level TRBC1/TRBC2 skew by T-cell subtype\n(dot size = fraction of sample T cells)"
        # xlim = (-7, 7) if xlim is None else xlim
        order_by = order_by or "max_clonality"

    else:
        raise ValueError("mode must be 'B' or 'T'")

    if metric not in df.columns:
        df[metric] = np.log2((df[left_col] + eps) / (df[right_col] + eps))

    max_frac = df[frac_col].max()
    df["dot_size"] = size_min + (size_max - size_min) * (
        df[frac_col] / max_frac
    )

    if order is None:
        if order_by == "median_clonality":
            order = (
                df.groupby(celltype_col, observed=True)[clonality_col]
                .median()
                .sort_values()
                .index
                .tolist()
            )
        elif order_by == "max_clonality":
            order = (
                df.groupby(celltype_col, observed=True)[clonality_col]
                .max()
                .sort_values()
                .index
                .tolist()
            )
        elif order_by == "median_metric":
            order = (
                df.groupby(celltype_col, observed=True)[metric]
                .median()
                .sort_values()
                .index
                .tolist()
            )
        elif order_by == "max_metric":
            order = (
                df.groupby(celltype_col, observed=True)[metric]
                .max()
                .sort_values()
                .index
                .tolist()
            )
        else:
            raise ValueError(
                "order_by must be one of: "
                "'median_clonality', 'max_clonality', "
                "'median_metric', 'max_metric'"
            )

    data = [
        df.loc[df[celltype_col] == ct, metric].dropna().values
        for ct in order
    ]

    if figsize is None:
        figsize = (6, max(5, 0.25 * len(order)))

    fig, ax = plt.subplots(figsize=figsize)

    ax.violinplot(
        data,
        vert=False,
        showmedians=True,
        showextrema=False,
    )

    rng = np.random.default_rng(random_state)

    for i, ct in enumerate(order, start=1):
        sub = df[df[celltype_col] == ct]
        y = i + rng.uniform(-jitter, jitter, size=len(sub))

        ax.scatter(
            sub[metric],
            y,
            s=sub["dot_size"],
            c=color_map.get(ct, "#999999"),
            alpha=alpha,
            linewidths=0,
        )

    ax.axvline(0, linestyle="--", linewidth=1)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order)
    ax.invert_yaxis()

    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax

    
# Plot clone size score distributions by cell type with sample-level dots,
# median markers, permutation-test thresholds, and FDR significance labels.

def plot_clone_size_score_violin(
    df_class,
    res=None,
    celltype_col="celltype",
    p_col="frac_kappa",
    metric="clone_size_score",
    frac_col="frac_cells_in_sample",
    order=None,
    order_by="median",
    xlim=(0, 1),
    size_min=2,
    size_max=20,
    jitter=0.3,
    alpha=0.5,
    dot_color=None,
    median_color="black",
    threshold_color="grey",
    random_state=0,
    figsize=None,
    title="sample-level largest clone size",
    xlabel="Clone size score",
    annotate_sig=True,
    q_cutoff=0.05,
):
    """
    Add sample-level clonality metrics to ``adata.obs``.

    Metrics are matched using sample and cell-type identifiers and copied
    into the observation table.

    Parameters
    ----------
    adata : AnnData
        AnnData object to annotate.
    df_class : pandas.DataFrame
        Data frame containing sample-level metrics.
    ...

    Returns
    -------
    AnnData
        Annotated AnnData object.
    """
    df = df_class.copy()

    if metric not in df.columns:
        clip_eps = 1e-6
        p = df[p_col].astype(float).clip(clip_eps, 1 - clip_eps)
        m = (
            df.groupby(celltype_col, observed=True)[p_col]
            .transform("median")
            .astype(float)
            .clip(clip_eps, 1 - clip_eps)
        )

        dist = (p - m).abs()
        max_dist = np.maximum(m, 1 - m)
        df[metric] = (dist / max_dist).clip(0, 1)

    max_frac = df[frac_col].max()
    df["dot_size"] = size_min + (size_max - size_min) * (
        df[frac_col] / max_frac
    )

    if order is None:
        if order_by == "median":
            order = (
                df.groupby(celltype_col, observed=True)[metric]
                .median()
                .sort_values()
                .index
                .tolist()
            )
        elif order_by == "mean":
            order = (
                df.groupby(celltype_col, observed=True)[metric]
                .mean()
                .sort_values()
                .index
                .tolist()
            )
        else:
            order = sorted(df[celltype_col].dropna().unique())

    data = [
        df.loc[df[celltype_col] == ct, metric].dropna().values
        for ct in order
    ]

    if figsize is None:
        figsize = (6, max(3, 0.25 * len(order)))

    fig, ax = plt.subplots(figsize=figsize)

    ax.violinplot(
        data,
        vert=False,
        showmedians=False,
        showextrema=False,
    )

    medians = [
        np.median(df.loc[df[celltype_col] == ct, metric].dropna().values)
        for ct in order
    ]

    ax.scatter(
        medians,
        range(1, len(order) + 1),
        color=median_color,
        s=10,
        zorder=3,
    )

    rng = np.random.default_rng(random_state)

    for i, ct in enumerate(order, start=1):
        sub = df[df[celltype_col] == ct]
        y = i + rng.uniform(-jitter, jitter, size=len(sub))

        ax.scatter(
            sub[metric].values,
            y,
            s=sub["dot_size"].values,
            c=dot_color,
            alpha=alpha,
            linewidths=0,
        )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlim(xlim)

    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title)

    if res is not None:
        res2 = res.copy()

        if "q_perm" not in res2.columns:
            res2["q_perm"] = multipletests(
                res2["p_perm"].values,
                method="fdr_bh",
            )[1]

        res2["significant"] = res2["q_perm"] < q_cutoff

        q_map = res2.set_index("celltype")["q_perm"]
        pct_map = res2.set_index("celltype")["pct_exceeding"]
        thr_map = res2.set_index("celltype")["threshold"]

        thr_vals = res2["threshold"].dropna().unique()

        if len(thr_vals) == 1:
            thr = float(thr_vals[0])

            xmin, xmax = ax.get_xlim()
            ax.set_xlim(min(xmin, 0), max(xmax, thr * 1.05))

            ax.axvline(
                thr,
                linestyle="--",
                linewidth=1,
                color=threshold_color,
            )

        else:
            for i, ct in enumerate(order, start=1):
                if ct in thr_map.index and np.isfinite(thr_map.loc[ct]):
                    ax.scatter(
                        [thr_map.loc[ct]],
                        [i],
                        marker="|",
                        s=200,
                        color=threshold_color,
                        zorder=4,
                    )

        if annotate_sig:
            def q_to_stars(q):
                if q < 1e-4:
                    return "****"
                elif q < 1e-3:
                    return "***"
                elif q < 1e-2:
                    return "**"
                elif q < 0.05:
                    return "*"
                return ""

            for i, ct in enumerate(order, start=1):
                if ct in q_map.index and q_map.loc[ct] < q_cutoff:
                    stars = q_to_stars(float(q_map.loc[ct]))
                    pct = float(pct_map.loc[ct])

                    ax.text(
                        1.02,
                        i,
                        f"{stars}  {pct:.1f}%",
                        va="center",
                        ha="left",
                        fontsize=10,
                        transform=ax.get_yaxis_transform(),
                        clip_on=False,
                    )

    plt.tight_layout()
    return fig, ax


# Add sample × celltype clonality/skew metrics from df_class into adata.obs.

def add_df_class_to_adata_obs(
    adata,
    df_class,
    sample_col="sample_id",
    celltype_col="celltype_B_v2",
    cols_to_add=None,
    prefix=""
):
    """
    Add sample-level clonality metrics to ``adata.obs``.

    Metrics are matched using sample and cell-type identifiers and copied
    into the observation table.

    Parameters
    ----------
    adata : AnnData
        AnnData object to annotate.
    df_class : pandas.DataFrame
        Data frame containing sample-level metrics.
    ...

    Returns
    -------
    AnnData
        Annotated AnnData object.
    """
    if cols_to_add is None:
        cols_to_add = [
            "kappa",
            "lambda",
            "frac_kappa",
            "KL_class_log2_ratio_scaled",
            "clonality",
            "clone_size_score",
        ]

    key_cols = [sample_col, celltype_col]

    metric_df = (
        df_class[key_cols + cols_to_add]
        .drop_duplicates(subset=key_cols)
        .copy()
    )

    metric_df[sample_col] = metric_df[sample_col].astype(str)
    metric_df[celltype_col] = metric_df[celltype_col].astype(str)

    obs = adata.obs.copy()
    obs[sample_col] = obs[sample_col].astype(str)
    obs[celltype_col] = obs[celltype_col].astype(str)

    obs = obs.merge(
        metric_df,
        on=key_cols,
        how="left",
        suffixes=("", "_from_df_class"),
    )

    obs.index = adata.obs.index

    for col in cols_to_add:
        new_col = f"{prefix}{col}"
        source_col = col if col in obs.columns else f"{col}_from_df_class"
        adata.obs[new_col] = obs[source_col].values

    return adata


# Select top and/or bottom clonal samples and return
# a subsetted AnnData object with ranking annotations.

def subset_top_bottom_clonal_samples(
    df_clono,
    adata,
    sample_col="sample_id",
    celltype_col="cell_type",
    score_col="clone_size_score",
    # --- eligibility filter ---
    frac_col="frac_cells_in_sample",
    frac_celltype=None,
    frac_percentile=0.95,
    # --- sample score aggregation ---
    score_agg="max",                 # "max" or "median"
    # --- selection modes ---
    mode="top_bottom",               # "top_bottom" or "range"
    top_n=10,
    bottom_n=10,
    score_range=None,                # (low, high), e.g. (0.4, 0.6)
    n_from_range=None,               # optional cap within range
    random_state=0,
):
    """
    Select samples with the highest, lowest, or intermediate clonality.

    Samples are ranked using an aggregated clone-size score and returned
    together with a subsetted AnnData object containing rank
    annotations.

    Parameters
    ----------
    df_clono : pandas.DataFrame
        Sample-level clonality metrics.
    adata : AnnData
        AnnData object to subset.
    ...

    Returns
    -------
    dict
        Dictionary containing selected samples, rankings, scores, and
        the subsetted AnnData object.
    """
    rng = np.random.default_rng(random_state)

    if score_col not in df_clono.columns:
        raise ValueError(f"{score_col} not in df_clono")

    # ---- sample-level score ----
    if score_agg == "max":
        sample_score = df_clono[df_clono[celltype_col]==frac_celltype].groupby(sample_col, observed=True)[score_col].max()
    elif score_agg == "median":
        sample_score = df_clono[df_clono[celltype_col]==frac_celltype].groupby(sample_col, observed=True)[score_col].median()
    else:
        raise ValueError("score_agg must be 'max' or 'median'")

    eligible = sample_score.index

    # ---- optional: filter by high fraction of a specific celltype ----
    sample_frac = None
    frac_thresh = None

    if frac_celltype is not None:
        sub = df_clono[df_clono[celltype_col] == frac_celltype][
            [sample_col, frac_col]
        ].dropna()
        if sub.empty:
            raise ValueError(f"No rows for celltype {frac_celltype}")

        sample_frac = sub.groupby(sample_col, observed=True)[frac_col].max()
        frac_thresh = float(np.quantile(sample_frac.values, frac_percentile))
        eligible = eligible.intersection(
            sample_frac[sample_frac >= frac_thresh].index
        )

    sample_score_elig = sample_score.loc[eligible].sort_values(ascending=False)
    if sample_score_elig.empty:
        raise ValueError("No eligible samples after filtering")

    # ---- selection ----
    if mode == "top_bottom":
        top_samples = sample_score_elig.head(top_n).index.tolist()
        bottom_samples = sample_score_elig.tail(bottom_n).index.tolist()
        selected = top_samples + bottom_samples

        rank_map = {d: "top" for d in top_samples}
        rank_map.update({d: "bottom" for d in bottom_samples})

    elif mode == "range":
        if score_range is None or len(score_range) != 2:
            raise ValueError("mode='range' requires score_range=(low, high)")

        lo, hi = score_range
        in_range = sample_score_elig[
            (sample_score_elig >= lo) & (sample_score_elig <= hi)
        ]

        if in_range.empty:
            raise ValueError("No samples in specified score_range")

        if n_from_range is not None and len(in_range) > n_from_range:
            in_range = in_range.sample(n_from_range, random_state=random_state)

        selected = in_range.index.tolist()
        rank_map = {d: "range" for d in selected}

        top_samples = []
        bottom_samples = []

    else:
        raise ValueError("mode must be 'top_bottom' or 'range'")

    # ---- subset adata ----
    mask = adata.obs[sample_col].astype(str).isin(map(str, selected))
    adata_sel = adata[mask].copy()

    # ---- annotate obs ----
    adata_sel.obs["sample_rank_class"] = pd.Categorical(
        adata_sel.obs[sample_col]
        .astype(str)          # avoid categorical issues
        .map(rank_map)
        .fillna("other"),
        categories=["top", "bottom", "range", "other"],
    )

    adata_sel.obs["sample_rank_score"] = adata_sel.obs[sample_col].map(sample_score)

    if sample_frac is not None:
        adata_sel.obs[f"sample_frac_{frac_celltype}"] = (
            adata_sel.obs[sample_col].map(sample_frac)
        )

    return {
        "selected_samples": selected,
        "top_samples": top_samples,
        "bottom_samples": bottom_samples,
        "sample_scores": sample_score_elig,
        "frac_threshold": frac_thresh,
        "adata_selected": adata_sel,
    }


# Detect sample-specific clonal populations using Leiden clustering
# and receptor-chain purity within clusters.

def leiden_chain_clones_sampleaware(
    adata,
    class_col,
    valid_states,
    sample_col="sample_uid_tpd",
    use_rep="X_pca",
    n_neighbors=30,
    resolution=0.8,
    purity_cutoff=0.8,
    key_added="clone_call",
    ignore_ambiguous=True,
    min_cells_in_cluster=10,
    random_state=0,
):
    """
    Detect clonal populations independently within each sample.

    Cells are clustered using Leiden, and clusters are labelled as
    clonal when a receptor-chain state exceeds the specified purity
    threshold.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing cells.
    class_col : str
        Column containing receptor-chain classifications.
    valid_states : sequence of str
        States considered valid for clone calling.
    ...

    Returns
    -------
    AnnData
        AnnData object annotated with clone assignments and cluster
        summaries.
    """
    adata.obs[key_added] = "polyclonal"
    adata.obs[key_added + "_id"] = "polyclonal"

    summaries = []

    for sample, idx in adata.obs.groupby(sample_col, observed=True).groups.items():
        idx = list(idx)
        if len(idx) < min_cells_in_cluster:
            continue

        sub = adata[idx].copy()

        # sample-specific neighbors/leiden
        sc.pp.neighbors(sub, n_neighbors=n_neighbors, use_rep=use_rep, random_state=random_state)
        sc.tl.leiden(sub, resolution=resolution, key_added="__leiden_tmp", random_state=random_state)

        clusters = sub.obs["__leiden_tmp"].astype(str)
        chain = sub.obs[class_col].astype(str)

        for clust, cidx in clusters.groupby(clusters).groups.items():
            # IMPORTANT: cidx are obs_names (strings)
            parent_cells = pd.Index(cidx)

            n_cells = len(parent_cells)
            if n_cells < min_cells_in_cluster:
                call = "polyclonal"
                major_state = None
                major_frac = np.nan
            else:
                x = chain.loc[parent_cells]
                if ignore_ambiguous:
                    x = x[x.isin(valid_states)]

                if x.empty:
                    call = "polyclonal"
                    major_state = None
                    major_frac = np.nan
                else:
                    frac = x.value_counts(normalize=True)
                    major_state = frac.index[0]
                    major_frac = float(frac.iloc[0])
                    call = f"clone_{major_state}" if (major_state in valid_states and major_frac >= purity_cutoff) else "polyclonal"

            clone_id = f"{sample}__{clust}" if call.startswith("clone_") else "polyclonal"

            # write back to parent adata
            adata.obs.loc[parent_cells, key_added] = call
            adata.obs.loc[parent_cells, key_added + "_id"] = clone_id

            summaries.append({
                "sample": sample,
                "cluster": clust,
                "n_cells": n_cells,
                "call": call,
                "purity": major_frac,
                "majority_state": major_state,
            })

    adata.obs[key_added] = pd.Categorical(adata.obs[key_added])
    adata.obs[key_added + "_id"] = pd.Categorical(adata.obs[key_added + "_id"])
    adata.uns[key_added + "_summary"] = pd.DataFrame(summaries)

    return adata


    
