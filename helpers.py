import streamlit as st
import matplotlib.pyplot as plt


def write_paper_table(data, n_words=True, distance=True):
  """
  Writes a markdown table of papers and their titles.

  :param data:
  :param n_words: bool; if True, add column with abstract word count.
  """
  table = f"""
  |Rank|Title|Value|{"# words|"*n_words}{"Similarity|"*distance}
  |--|--|--|--|
  """
  for i, el in enumerate(data):
    line = f"""|{i+1}|**{el["title"]}**|£{el["value"]:,}|"""
    if n_words:
      line += f"""{str(el["n_words"])}|"""
    if distance:
      line += f"""{str(round(el["distance"], 2))}"""
    line = f"""{line}
            """
    table += line

  st.markdown(table)

def sparkline(data, figsize=(4, 0.25), **kwargs):
  """
  Creates a sparkline from a list of ints or floats.

  :param data: np.array, series, or list of data to plot
  :param figsize: matplotlib figsize of plot
  :param **kwargs: other keyword arguments for plt.subplots
  """

  data = list(data)
 
  fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)
  x = [i for i in range(len(data))]
  ax.plot(data)
  ax.fill_between(x, data, len(data)*[min(data)], alpha=0.1)
#   ax.set_axis_off()
  ax.xaxis.set_ticks([min(x), max(x)])
  ax.xaxis.set_ticklabels(["2015", "2021"])
  ax.yaxis.set_visible(False)
  artists = ax.get_children()
  artists.remove(ax.yaxis)
  ax.tick_params(axis=u'both', which=u'both',length=0)
  ticklabels = ax.get_xticklabels()
  # set the alignment for outer ticklabels
  ticklabels[0].set_ha("left")
  ticklabels[-1].set_ha("right")
  ax.set_frame_on(False)
  ax.axis('tight')
  
  fig.set_tight_layout(True)
  fig.subplots_adjust(left=0)

  return fig