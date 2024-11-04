from scipy.stats import zscore
from statsmodels.formula import api as smf

from plots_2D import multiseries_scatter as scat, sliding_window_percentile as swp


def linear_analysis(df, x, y, output_folder, covs=(), normalise=True, show_ims=True):
    print(f'Analysing {y} against {x}')
    data = [[[1], df.dropna(subset=[x,y])]]
    swp(data, x, y, show_ims=show_ims, save_to=output_folder + f'{x} vs {y} - sliding window.svg')
    scat(data, x, y, show_ims=show_ims, save_file=output_folder + f'{x} vs {y} - scatter.svg')

    formula = f'{y} ~ {x}' + ''.join([' + ' + c for c in covs])
    d = df.dropna(subset=[x, y])
    d[x], d[y] = d[x].astype(float), d[y].astype(float)

    if normalise:
        for c in covs + [x]:
            d[c] = d[[c]].dropna().apply(zscore)

    mod = smf.ols(formula=formula, data=d)
    result = mod.fit()
    with open('_'.join([output_folder, formula.replace('*', 'x'), '.txt']), 'w') as f:
        print(result.summary2(), file=f)

    return result
