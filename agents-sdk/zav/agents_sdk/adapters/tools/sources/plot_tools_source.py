from typing import Any, Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.async_wrapper import asyncify
from zav.agents_sdk.adapters.tools.sources.plotting import (
    PLOT_PALETTE,
    create_plot_image_uri,
    style_axes,
)
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, exclude_fields, streamable

MAX_PLOT_SERIES = 10
MAX_PLOT_POINTS = 10000

PlotType = Literal[
    "line",
    "area",
    "bar",
    "stacked_bar",
    "horizontal_bar",
    "pie",
    "donut",
    "scatter",
    "histogram",
    "box",
]

# Types that plot series against shared x-axis categories/values.
_XY_TYPES = ("line", "area", "bar", "stacked_bar", "scatter")
# Types that need x_values (categories/labels); histogram/box summarize raw values.
_CATEGORY_TYPES = _XY_TYPES + ("horizontal_bar", "pie", "donut")
_SINGLE_SERIES_TYPES = ("pie", "donut")

_SYSTEM_PROMPT_SECTION = """\
## Plotting

You can visualize data for the user with the `create_plot` tool by passing \
the data points directly.

- Use it to plot data you computed or gathered yourself (search results, \
aggregations, comparisons), without needing a loaded dataframe.
- Chart types: `line` and `area` for trends; `bar`, `stacked_bar`, and \
`horizontal_bar` for comparisons and rankings; `pie` and `donut` for \
composition/share; `scatter` for correlation; `histogram` and `box` for \
distributions.
- `create_plot` returns a `plot_id`. Embed each chart inline as a Markdown \
image where it belongs in your answer, using the id as the source: \
`![<short caption describing the chart>](plot:<plot_id>)`. Write the \
surrounding text and image as you go — do not batch all plots at the end.
- Keep the data small and meaningful: at most {max_series} series and \
{max_points} total points.\
""".format(max_series=MAX_PLOT_SERIES, max_points=MAX_PLOT_POINTS)


class PlotToolsSourceConfiguration(BaseModel):
    enabled: bool = False


class PlotSeries(BaseModel):
    name: str
    values: List[float]


class PlotToolsSource(ToolsSource):

    source_name = "plot_tools"

    def __init__(
        self,
        plot_tools_source_configuration: PlotToolsSourceConfiguration = (
            PlotToolsSourceConfiguration()
        ),
    ):
        self.enabled = plot_tools_source_configuration.enabled
        self.__plot_count = 0

    async def get_tools(self) -> List[Tool]:
        return [
            Tool.from_callable(
                name="create_plot",
                executable=self.create_plot,
            ),
        ]

    async def to_prompt(self) -> str:
        if not self.enabled:
            return ""
        return _SYSTEM_PROMPT_SECTION

    @streamable(
        running_text=(
            "Creating {{ plot_type }} plot" "{% if title %}: {{ title }}{% endif %}..."
        ),
        completed_text=(
            "Created {{ plot_type }} plot" "{% if title %}: {{ title }}{% endif %}."
        ),
        llm_response_transform=exclude_fields("image_uri"),
    )
    async def create_plot(
        self,
        plot_type: PlotType,
        series: List[PlotSeries],
        x_values: Optional[List[Union[str, float]]] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Plot data passed directly as values and show the image to the user.

        Args:
            plot_type: 'line', 'area', 'bar', 'stacked_bar', 'horizontal_bar',
                'pie', 'donut', 'scatter', 'histogram', or 'box'.
            series: Named series of numeric values. For 'histogram' and 'box'
                each series holds the raw values to summarize; 'pie'/'donut'
                take exactly one series of wedge sizes; otherwise each series
                holds one value per x_value.
            x_values: X-axis values / category labels. Required for everything
                except 'histogram' and 'box'; for 'pie'/'donut' these label the
                wedges. Must have the same length as every series.
            title: Plot title.
            x_label: X-axis label.
            y_label: Y-axis label.

        Returns:
            On success, a dict with the plot ``description``, a ``plot_id``, and
            the rendered ``image_uri``. Embed the chart inline as a Markdown
            image where it belongs: ``![<caption>](plot:<plot_id>)``. On
            failure, an error dict.
        """
        try:
            validation = self.__validate_plot_input(plot_type, series, x_values)
            if validation["status"] == "error":
                return validation

            fig, ax = plt.subplots(figsize=(10, 6))
            style_axes(ax)
            if plot_type in _SINGLE_SERIES_TYPES:
                await self.__draw_pie(ax, series, x_values, donut=plot_type == "donut")
            elif plot_type == "histogram":
                await self.__draw_histogram(ax, series, x_label, y_label)
            elif plot_type == "box":
                await self.__draw_box(ax, series, x_label, y_label)
            elif plot_type == "horizontal_bar":
                await self.__draw_horizontal_bar(ax, series, x_values, x_label, y_label)
            else:
                await self.__draw_xy(ax, plot_type, series, x_values, x_label, y_label)

            if title:
                ax.set_title(title)
            if len(series) > 1 and plot_type not in ("pie", "donut", "box"):
                # Place the legend outside the axes so it never covers bars/lines
                # (the default loc="best" still lands on data when a tall series
                # fills a corner). savefig's bbox_inches="tight" grows the image
                # to include it, so it is never clipped.
                ax.legend(
                    loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0
                )
            plt.tight_layout()
            image_uri = await create_plot_image_uri(fig)

            series_names = ", ".join(plot_series.name for plot_series in series)
            plot_description = f"Created {plot_type} plot of '{series_names}'"
            if title:
                plot_description += f" (titled: {title})"
            self.__plot_count += 1
            plot_id = f"p{self.__plot_count}"
            return {
                "status": "ok",
                "description": plot_description,
                "plot_id": plot_id,
                "image_uri": image_uri,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def __draw_histogram(
        self,
        ax,
        series: List[PlotSeries],
        x_label: Optional[str],
        y_label: Optional[str],
    ) -> None:
        for plot_series in series:
            await asyncify(ax.hist)(
                plot_series.values,
                bins=30,
                alpha=0.7,
                label=plot_series.name,
            )
        ax.set_ylabel(y_label or "Frequency")
        if x_label:
            ax.set_xlabel(x_label)

    async def __draw_box(
        self,
        ax,
        series: List[PlotSeries],
        x_label: Optional[str],
        y_label: Optional[str],
    ) -> None:
        await asyncify(ax.boxplot)(
            [plot_series.values for plot_series in series],
            tick_labels=[plot_series.name for plot_series in series],
        )
        if y_label:
            ax.set_ylabel(y_label)
        if x_label:
            ax.set_xlabel(x_label)

    async def __draw_pie(
        self,
        ax,
        series: List[PlotSeries],
        x_values: Optional[List[Union[str, float]]],
        donut: bool,
    ) -> None:
        labels = [str(x) for x in x_values] if x_values else [series[0].name]
        wedgeprops = {"edgecolor": "white", "linewidth": 1}
        if donut:
            wedgeprops["width"] = 0.45
        await asyncify(ax.pie)(
            series[0].values,
            labels=labels,
            autopct="%1.1f%%",
            colors=PLOT_PALETTE,
            wedgeprops=wedgeprops,
        )
        # A pie has no axes furniture; drop the spines/ticks/grid style_axes set.
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_aspect("equal")

    async def __draw_horizontal_bar(
        self,
        ax,
        series: List[PlotSeries],
        x_values: Optional[List[Union[str, float]]],
        x_label: Optional[str],
        y_label: Optional[str],
    ) -> None:
        assert x_values is not None
        labels = [str(x) for x in x_values]
        y_positions = list(range(len(labels)))
        bar_height = 0.8 / len(series)
        for index, plot_series in enumerate(series):
            offset = (index - len(series) / 2 + 0.5) * bar_height
            await asyncify(ax.barh)(
                [y + offset for y in y_positions],
                plot_series.values,
                height=bar_height,
                label=plot_series.name,
            )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        # The value axis is horizontal here, so grid on x instead of y.
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#e7e9ee", linewidth=0.8)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

    async def __draw_xy(
        self,
        ax,
        plot_type: str,
        series: List[PlotSeries],
        x_values: Optional[List[Union[str, float]]],
        x_label: Optional[str],
        y_label: Optional[str],
    ) -> None:
        assert x_values is not None
        is_categorical = any(not isinstance(x, (int, float)) for x in x_values)
        x_positions: List[Any] = (
            list(range(len(x_values))) if is_categorical else list(x_values)
        )
        if plot_type == "bar":
            bar_width = 0.8 / len(series)
            for index, plot_series in enumerate(series):
                offset = (index - len(series) / 2 + 0.5) * bar_width
                await asyncify(ax.bar)(
                    [x + offset for x in x_positions],
                    plot_series.values,
                    width=bar_width,
                    label=plot_series.name,
                )
        elif plot_type == "stacked_bar":
            bottom = [0.0] * len(x_positions)
            for plot_series in series:
                await asyncify(ax.bar)(
                    x_positions,
                    plot_series.values,
                    bottom=bottom,
                    label=plot_series.name,
                )
                bottom = [b + v for b, v in zip(bottom, plot_series.values)]
        elif plot_type == "area":
            # No explicit colors: stackplot falls back to the Axes prop-cycle,
            # which style_axes set to PLOT_PALETTE (and cycles for many series).
            await asyncify(ax.stackplot)(
                x_positions,
                *[plot_series.values for plot_series in series],
                labels=[plot_series.name for plot_series in series],
                alpha=0.85,
            )
        else:
            for plot_series in series:
                if plot_type == "line":
                    await asyncify(ax.plot)(
                        x_positions,
                        plot_series.values,
                        label=plot_series.name,
                        marker="o",
                        markersize=3,
                    )
                else:
                    await asyncify(ax.scatter)(
                        x_positions,
                        plot_series.values,
                        label=plot_series.name,
                        alpha=0.6,
                        s=30,
                    )
        if is_categorical:
            labels = [str(x) for x in x_values]
            # Only slant labels when they're long enough to overlap; short ones
            # (Jan, Organic) stay horizontal and uncluttered.
            rotate_labels = any(len(label) > 8 for label in labels)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                labels,
                rotation=45 if rotate_labels else 0,
                ha="right" if rotate_labels else "center",
            )
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

    def __validate_plot_input(
        self,
        plot_type: str,
        series: List[PlotSeries],
        x_values: Optional[List[Union[str, float]]],
    ) -> Dict[str, Any]:
        if not series:
            return {"status": "error", "message": "At least one series is required."}
        if len(series) > MAX_PLOT_SERIES:
            return {
                "status": "error",
                "message": f"At most {MAX_PLOT_SERIES} series are supported.",
            }
        total_points = sum(len(plot_series.values) for plot_series in series)
        if total_points > MAX_PLOT_POINTS:
            return {
                "status": "error",
                "message": (
                    f"Too many data points ({total_points})," f" max {MAX_PLOT_POINTS}."
                ),
            }
        if any(not plot_series.values for plot_series in series):
            return {"status": "error", "message": "Every series needs values."}
        if plot_type in _SINGLE_SERIES_TYPES and len(series) != 1:
            return {
                "status": "error",
                "message": f"'{plot_type}' supports exactly one series.",
            }
        if plot_type in _CATEGORY_TYPES:
            if not x_values:
                return {
                    "status": "error",
                    "message": f"x_values is required for plot type '{plot_type}'.",
                }
            mismatched = [
                plot_series.name
                for plot_series in series
                if len(plot_series.values) != len(x_values)
            ]
            if mismatched:
                return {
                    "status": "error",
                    "message": (
                        "Series must have the same length as x_values:"
                        f" {', '.join(mismatched)}"
                    ),
                }
        if plot_type in _SINGLE_SERIES_TYPES and any(v < 0 for v in series[0].values):
            return {
                "status": "error",
                "message": f"'{plot_type}' values must be non-negative.",
            }
        return {"status": "ok"}


class PlotToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        plot_tools_source_configuration: PlotToolsSourceConfiguration = (
            PlotToolsSourceConfiguration()
        ),
    ) -> PlotToolsSource:
        return PlotToolsSource(
            plot_tools_source_configuration=plot_tools_source_configuration,
        )
