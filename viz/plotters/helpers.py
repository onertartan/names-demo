import pandas as pd
import streamlit


def year_ticks(years) -> list[int]:


    years = sorted( years)
    if len(years) <= 12:
        return years

    start, end = years[0], years[-1]
    span = end - start
    if span > 80:
        step = 10
    elif span > 35:
        step = 5
    elif span > 15:
        step = 2
    else:
        step = 1

    first_tick = ((start + step - 1) // step) * step
    ticks = list(range(first_tick, end + 1, step))
    if not ticks or ticks[0] != start:
        ticks = [start] + ticks
    return sorted(set(ticks))