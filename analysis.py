import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def fix_model_names(x):
    return (
        x.replace("_hidden", "")
        .replace("_lit_gpt", "")
        .replace("_reproduce_eval", "")
        .replace("datta0_neurips_submission_1", "datta0_neurips_submission")
        .replace("tvergho_lm_neurips_train", "tvergho_llm_neurips_train")
        .replace(
            "mrigankramanllm_compa100_submissionsa100_1st_submission",
            "mrigankraman_llm_comp_a100_submissions_a100_1st_submission",
        )
    )


def calculate_win_rate(series):
    assert len(series) > 1, "no meaning for a win rate with only one object"

    def win_rate(x):
        win_count = sum(1 for value in series if x > value)
        return win_count / (len(series) - 1)

    return series.transform(win_rate)


def main():
    res_dir = "results_files"
    paths = [f"{res_dir}/{file}" for file in os.listdir(res_dir)]

    res = []

    for path in paths:
        df = pd.read_json(path)

        # Expand the dictionary in column 1 into a separate DataFrame
        rows = []
        for idx, row in df.iterrows():
            model = row[0]
            results = row[1]
            for dataset, score in results.items():
                rows.append({"model": model, "scenario": dataset, "score": score})

        # Create a new DataFrame
        results_df = pd.DataFrame(rows)
        results_df["file"] = path.split("/")[-1]

        res.append(results_df)

    res = pd.concat(res)
    res["model"] = res["model"].apply(fix_model_names)

    res["track"] = res["file"].apply(lambda x: x.split("_")[0])
    res["subset"] = res["file"].apply(lambda x: x.split("_")[1])

    res["wr"] = res.groupby(["scenario"])["score"].transform(calculate_win_rate)

    print(res[["model", "subset"]].drop_duplicates().groupby("subset").count())
    models_in_full = res.query('subset=="full"')["model"].unique()

    # counts
    print(
        res[["model", "subset", "track"]]
        .query("model in @models_in_full")
        .drop_duplicates()
        .groupby(["subset", "track"])
        .count()
    )

    res["rank"] = res.groupby(["subset", "track", "scenario"])["score"].rank()

    sns.catplot(
        data=res.groupby(["model", "subset", "track"]).agg(
            {"score": "mean", "wr": "mean"}
        ),
        x="score",
        y="wr",
        hue="track",
        col="subset",
    )
    plt.show(block=True)

    # sns.regplot(

    sns.set(style="white", font_scale=1.5)

    plot_df = (
        res.query('subset=="open" or subset=="hidden"')
        .groupby(["model", "subset", "track"])
        .agg({"wr": "mean", "score": "mean"})
        .reset_index()
        .pivot(index=["model", "track"], columns="subset", values="wr")
        .reset_index()[["hidden", "open", "track"]]
        .rename(
            columns={
                "hidden": "Hidden Evaluation Set MWR",
                "open": "Open Evaluation Set MWR",
            }
        )
    )

    plot_df.query("track=='4090'").dropna()[
        ["Hidden Evaluation Set MWR", "Open Evaluation Set MWR"]
    ].corr()
    plot_df.query("track=='A100'").dropna()[
        ["Hidden Evaluation Set MWR", "Open Evaluation Set MWR"]
    ].corr()

    sns.lmplot(
        data=plot_df,
        x="Hidden Evaluation Set MWR",
        y="Open Evaluation Set MWR",
        hue="track",
    )
    plt.savefig("open_hidden_corr.pdf")
    plt.show()

    plot_df = (
        res.query('subset=="hidden" or subset=="full"')
        .groupby(["model", "subset", "track"])
        .agg({"wr": "mean", "score": "mean"})
        .reset_index()
        .pivot(index=["model", "track"], columns="subset", values="wr")
        .reset_index()[["full", "hidden", "track"]]
        .rename(
            columns={
                "full": "Full Evaluation Set MWR",
                "hidden": "Hidden Evaluation Set MWR",
            }
        )
    )

    plot_df.query("track=='4090'").dropna()[
        ["Hidden Evaluation Set MWR", "Full Evaluation Set MWR"]
    ].corr()
    plot_df.query("track=='A100'").dropna()[
        ["Hidden Evaluation Set MWR", "Full Evaluation Set MWR"]
    ].corr()

    sns.lmplot(
        data=plot_df,
        x="Hidden Evaluation Set MWR",
        y="Full Evaluation Set MWR",
        hue="track",
    )
    plt.savefig("full_hidden_corr.pdf")
    plt.show()

    ######################################################################
    ######################################################################
    # rank agree betweebn scenarios

    # .groupby(["model", "subset", "track","scenario"])
    # .agg({"wr": "mean", "score": "mean"})
    # .reset_index()
    # .reset_index()
    # .rename(
    #     columns={
    #         "full": "Full Evaluation Set MWR",
    #         "hidden": "Hidden Evaluation Set MWR",
    #     }
    # )

    plot_df = (
        res.replace("hidden", "open_hidden")
        .replace("open", "open_hidden")
        .dropna()
        .drop_duplicates(subset=["model", "track", "scenario", "subset"])
        # .query('(subset=="open_hidden" or subset=="full")')
        .pivot(index=["model", "track", "scenario"], columns="subset", values="score")
        # .dropna()
        .reset_index()
        .query("full>0 and open_hidden>0")
    )

    for f in ["Robustness", "Fairness", "Win Rate", "race", "gender"]:
        mask = ~plot_df["scenario"].str.contains(f, na=False)
        # Use the mask inside the query method
        plot_df = plot_df.query("@mask")



    scenarios_to_take = [
        # "BBQ - EM",
        # "MATH (chain-of-thoughts) - Equivalent (chain of thought)",
        "MMLU - EM",
        "TruthfulQA - EM",
        # "corr2cause - EM",
        # "ethics_commonsense - EM",
        # "ethics_deontology - EM",
        "ethics_justice - EM",
        # "ethics_utilitarianism - EM",
        # "ethics_virtue - EM",
        # "sam_sum - ROUGE-2",
        "BIG-bench - EM",
        # "GSM8K - EM",
    ]

    sns.set(font_scale=1.6, style='white')

    g = sns.lmplot(
        data=plot_df.query("scenario in @scenarios_to_take").rename(columns={'full':'Full eval set','open_hidden':'Open+Hidden sets'}),
        x="Full eval set",
        y="Open+Hidden sets",
        col="scenario",
        col_wrap=4,
        hue="track",
        sharex=False,
        sharey=False,
        ci=None,
    )
    g = g.set_titles("{col_name}")

    plt.savefig("figures/scenario_wise_full_open+hidden_corr.pdf")
    plt.show()

    g = sns.lmplot(
        data=plot_df,
        x="full",
        y="open_hidden",
        # col="scenario",
        # col_wrap=4,
        hue="track",
        sharex=False,
        sharey=False,
        ci=None,
    )
    g = g.set_titles("{col_name}")

    plt.savefig("figures/full_open+hidden_corr.pdf")
    plt.show()

    # ranks_plot
    df = (
        res.query('subset=="full" and track=="A100"')
        .drop_duplicates(["model", "scenario"])
        .groupby(["model", "scenario"])["wr"]
        .mean()
        .reset_index()
    )

    # Step 1: Identify the best model based on overall average score
    best_models = df.groupby("model")["wr"].mean().nlargest(1).index

    # Step 2: Filter the DataFrame to include only the best models
    best_model_scores = df[df["model"].isin(best_models)]

    # Step 3: Group by scenario and calculate the average score for these best models
    scenario_performance = best_model_scores.groupby("scenario")["wr"].mean()

    # Step 4: Sort the scenarios by their performance and create a ranking
    sorted_scenarios = scenario_performance.sort_values(ascending=False)
    scenario_ranking = pd.Series(
        range(1, len(sorted_scenarios) + 1), index=sorted_scenarios.index
    )

    # Step 5: Merge the scenario ranking back to the original DataFrame
    df = df.join(scenario_ranking.rename("scenario_rank"), on="scenario")

    # Step 6: Sort the DataFrame based on the scenario rank
    sorted_df = df.sort_values(by="scenario_rank")

    for f in ["Robustness", "Fairness", "Win Rate", "race", "gender", "Score"]:
        mask = ~sorted_df["scenario"].str.contains(f, na=False)
        # Use the mask inside the query method
        sorted_df = sorted_df.query("@mask")

    sorted_df["scenario"] = sorted_df["scenario"].apply(
        lambda x: x.split(" - ")[0].split(" (")[0].lower().replace("_", " ").strip()
    )

    sorted_df["model"] = sorted_df["model"].apply(lambda x: x.split("_")[0])

    plt.figure(figsize=(8, 4))

    ax = sns.pointplot(
        data=sorted_df,
        x="scenario",
        hue="model",
        y="wr",
        legend=True,
    )
    plt.xticks(rotation=45)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # plt.subplots_adjust(0.2)
    plt.savefig("figures/full_set_ranks.pdf")

    plt.show(block=True)
    # .replace("open", "open_hidden")
    # .dropna()
    # .drop_duplicates(subset=["model", "track", "scenario", "subset"])
    # # .query('(subset=="open_hidden" or subset=="full")')
    # .pivot(index=["model", "track", "scenario"], columns="subset", values="score")
    # # .dropna()
    # .reset_index()
    # .query("full>0 and open_hidden>0")


if __name__ == "__main__":
    main()
