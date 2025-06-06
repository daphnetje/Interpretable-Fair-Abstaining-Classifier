# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import plotnine as p9
import textwrap

def wraping_func(text):
    return [textwrap.fill(wraped_text, 15) for wraped_text in text]

def visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, performance_measure_of_interest, y_axis_limits=(0, 1)):
    dataframes_split_by_sens_features = dict(tuple(averaged_performances.groupby("Sensitive Features")))
    for key, dataframe in dataframes_split_by_sens_features.items():
        plot_title = performance_measure_of_interest + " " + key
        plot = visualize_performance_across_groups_different_classification_types_with_conf_intervals(dataframe, performance_measure_of_interest, title=plot_title, y_axis_limits=y_axis_limits)
        # plot_path = path_to_save_figure + "\\" + plot_title + ".svg"
        # p9.ggsave(plot, filename=plot_path)
    return

def visualize_performance_across_groups_different_classification_types_with_conf_intervals(data, performance_measure_of_interest, title, y_axis_limits):
    plot = p9.ggplot(data=data, mapping=p9.aes(x="Classification Type", y=performance_measure_of_interest + ' mean', fill="Group")) + \
           p9.geom_col(stat='identity', position='dodge', width=0.86) + \
           p9.geom_errorbar(p9.aes(x='Classification Type', ymin=performance_measure_of_interest + ' ci_low', ymax= performance_measure_of_interest + ' ci_high', fill='Group'),
                         position=p9.position_dodge(0.86), width=0.4) + \
           p9.scale_y_continuous(limits=y_axis_limits) + \
           p9.scale_x_discrete(breaks=data['Classification Type'].unique().tolist(), labels=wraping_func) + \
           p9.theme(panel_grid_minor=p9.element_blank()) + \
           p9.ggtitle(title) + \
           p9.scale_fill_manual(values=["#F8766D", "#B79F00", "#00BA38", "#00BFC4", "#2176FF", "#F564E3"])


    print(plot)
    return plot