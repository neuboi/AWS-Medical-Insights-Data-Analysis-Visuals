"""

Medical Insights Data Analysis Tool
Date: 10-20-26

"""

import re
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as Figure
from matplotlib.widgets import CheckButtons, RangeSlider, Button, TextBox

from old.data_analysis import get_statistics
from old.data_analysis import get_entities_in_sliding_char_window as _lib_get_entities_in_char_window, is_entity_in_char_window

from wordcloud import WordCloud, STOPWORDS
HAVE_WORDCLOUD = True

WIDTH = 0.35
HIGHLIGHT = 'tab:orange'
BAR_COLOR = 'tab:blue'
DEFAULT_TOTAL_CHARS = 100_000

def create_variables(window=10, score=None):
    return get_statistics(window, score)

def get_entities_in_char_window(entities, character_window, score):
    return _lib_get_entities_in_char_window(entities, character_window, score)

def entites_to_variables(window_ents):
    categories = []
    scores_set = defaultdict(list)
    text_per_category = defaultdict(list)
    for ent in window_ents:
        cat = ent['category']
        categories.append(cat)
        scores_set[cat].append(ent['score'])
        text_per_category[cat].append(ent['text'])
    return categories, set(categories), dict(scores_set), dict(text_per_category)

def _in_range_mask(values, score_range):
    if score_range is None:
        return np.ones(values.shape, dtype=bool)
    lo, hi = score_range
    return (values >= lo) & (values <= hi)

def draw_category_plot(ax1, cats, counts_by_cat, score_set, score_range=None, 
                       count_ylim=None, score_ylim=None, width=WIDTH, focus=None,
                       title='Category Counts and Score Distributions'):
    """
    Draws a bar plot of counts by category and a boxplot of score distributions by category.
    """
    ax2 = ax1.twinx()  # Create a second y-axis for the score distributions
    ax1.set_title(title)

    if not cats:
        ax1.text(0.5, 0.5, 'No categories selected', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12)
        ax1.set_xticks([])
        ax2.set_xticks([])
        return ax2

    x = np.arange(len(cats))  # the label locations
    totals = np.array([counts_by_cat.get(cat, 0) for cat in cats], dtype=float)
    score_arrays = [np.array(score_set.get(cat, []), dtype=float).ravel() for cat in cats]
    kept = []
    for total, values in zip(totals, score_arrays):
        if values.size == 0:
            kept.append(total)
        else:
            kept.append(min(total, np.count_nonzero(_in_range_mask(values, score_range))))
    kept = np.array(kept, dtype=float)

    colors = [HIGHLIGHT if cat == focus else BAR_COLOR for cat in cats]

    ax1.bar(x - width/2, kept, width, color=colors, label='Counts')

    for xi, k, n in zip(x, kept, totals):
        ax1.text(xi - width/2, k, f'{int(k)}', ha='center', va='bottom', 
                 fontsize=8)

    ax1.set_ylabel('Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel('Categories')

    ax2.boxplot(
        [values[_in_range_mask(values, score_range)] for values in score_arrays],
        positions=x + width/2,
        widths=width,
        manage_ticks=True
    )

    ax2.set_ylabel('Scores')

    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=45, ha='right')
    ax1.set_xlim(-0.6, len(cats) - 0.4)

    if count_ylim is None:
        count_ylim = (0, max(1.0, totals.max() * 1.15))

    ax1.set_ylim(*count_ylim)

    if score_ylim is None:
        ax2.set_ylim(*score_ylim)

    return ax2

def word_frequencies(cats, text_per_category):
    counter = Counter()
    for cat in cats:
        for item in text_per_category.get(cat, []):
            for word in re.findall(r"\w[\w']*", str(item).lower()):
                if word not in STOPWORDS and len(word) > 1:
                    counter[word] += 1
    return counter

def render_word_image(counter):
    if not HAVE_WORDCLOUD or not counter:
        return None
    wc = WordCloud(width=600, height=650, background_color='white', max_words=100, stopwords=STOPWORDS)
    return wc.generate_from_frequencies(counter).to_array()

def draw_word_panel(ax, counter, image, title):
    ax.clear()
    ax.set_title(title, fontsize=10)
    if not counter:
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'No text for this selection', ha='center', va='center',
                transform=ax.transAxes)
    elif image is not None:
        ax.imshow(image, interpolation='bilinear')
        ax.set_axis_off()
    else:
        ax.set_axis_on()
        words, freqs = zip(*counter.most_common(20))
        y = np.arange(len(words))
        ax.barh(y, freqs, color='tab:green')
        ax.set_yticks(y)
        ax.set_yticklabels(words, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency')

def _score_bounds(all_cats, scores_set):
    arrays = [np.asarray(scores_set.get(c, []), dtype=float).ravel() for c in all_cats]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return 0.0, 1.0
    values = np.concatenate(arrays)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = 0.05 * (hi - lo) if hi > lo else 0.5
    return lo - pad, hi + pad

def _parse_int(text, name):
    try:
        return int(text.strip())
    except ValueError:
        raise ValueError(f'{name} must be a whole number')

def _parse_score(text):
    t = text.strip()
    if t == '' or t.lower() == 'none':
        return None
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    raise ValueError('Score must be a number or blank/None')

def _unpack(result, fallback_texts):
    if len(result) == 4:
        categories, categories_set, scores_set, texts = result
    else:
        categories, categories_set, scores_set = result
    return {'cats': list(categories_set),
            'counts': Counter(categories),
            'scores': scores_set,
            'texts': texts,
            'n_entities': len(categories)}

# =============================================================

def create_data_analysis_tool(score = None, text_per_category=None, initial=None, entites=None,
        character_window=1000, window_mode=False, debounce_ms=150,
        save_path='category_distribution_and_scores.png',
        wordcloud_path='category_wordcloud.png'):

    params = {'score': score, 'character_window': character_window, 'total_chars': 10_000}
    full = {}
    data = {}
    state = {
        'ax1': None,
        'ax2': None,
        'check': None,
        'check_ax': None,
        'score_slider': None,
        'char_slider': None,
        'score_bounds': (0.0, 1.0),
        'focus': None,
        'word_cache': {},
        'window_on': bool(window_mode)
    }

    fig = plt.figure(figsize=(16, 8.5))
    plot_rect = [0.05, 0.36, 0.42, 0.56]
    ax_words = fig.add_axes([0.56, 0.36, 0.21, 0.56])
    entity_text = fig.text(0.05, 0.965, '', fontsize=13, fontweight='bold')

    def load_full():
        full.clear()
        full.update(_unpack(create_variables(window=params['character_window'], score=params['score']), text_per_category))

    def load_window():
        start = int(state['char_slider'].val)
        if entites is not None:
            window_entites = get_entities_in_char_window(
                entites,
                params['character_window'],
                start,
                params['score']
            )
            view = _unpack(entites_to_variables(window_entites), None)
            view['n_entites'] = len(window_entites)

        return _unpack(create_variables(window=params['character_window'], score=params['score']), text_per_category)

    def load_view():
        view = load_window() if state['window_on'] else full
        extras = [c for c in view['cats'] if c not in full['cats']]
        data.update(
            cats=full['cats'] + extras,
            counts=view['counts'],
            scores=view['scores'],
            texts=view['texts'],
            n_entites=view['n_entities']
        )
        state['word_cache'].clear()

    def entites_in_score_range():
        selected_cats = selected()
        range = score_range()
        kept = 0
        for cat in data['cats']:
            if cat not in selected_cats:
                continue
            total = data['counts'].get(cat, 0)
            values = np.asarray(data['scores'].get(cat, []), dtype=float).ravel()
            if values.size == 0:
                kept += total
            else:
                kept += min(total, np.count_nonzero(_in_range_mask(values, range)))
        return int(kept)

    def entity_summary():
        n = data['n_entites']
        return (f'Total Unique Entities: {entites_in_score_range():,} / {n:,}')

    def window_label():
        if not state['window_on']:
            return 'all characters'
        start = int(state['char_slider'].val)
        return f"chars {start:,}-{start + params['character_window']:,}"

    def selected():
        check = state['check']
        if check is None:
            return []
        return [c for c, on in zip(data['cats'], check.get_status()) if on]

    def word_cats():
        sel = selected()
        if state['focus'] not in sel:
            state['focus'] = None
        return [state['focus']] if state['focus'] else sel

    def word_title(hint=True):
        base = (f"All Words in {state['focus']}" if state['focus'] else f'All Words in {len(selected())} selected categories')
        if not hint: return base
        return base + (' (click bar again to view all)' if state['focus'] else ' (click a bar to focus)')

    def word_data(cats):
        key = tuple(cats)
        if key not in state['word_cache']:
            counter = word_frequencies(cats, data['texts'])
            state['word_cache'][key] = (counter, render_word_image(counter))
        return state['word_cache'][key]

    def plot_title():
        return f'Category Counts and Score Distribution ({window_label()})'

    def count_ylim():
        top = max((data['counts'].get(c, 0) for c in data['cats']), default=0)
        return (0, max(1.0, top * 1.15))

    def score_range():
        s = state['score_slider']
        return tuple(s.val) if s is not None else None

    def redraw_plot():
        for key in ('ax2', 'ax1'):
            if state[key] is not None:
                state[key].remove()
        ax1 = fig.add_axes(plot_rect)
        word_cats()
        ax2 = draw_category_plot(
            ax1,
            selected(),
            data['counts'],
            data['scores'],
            score_range=score_range(),
            count_ylim=count_ylim(),
            score_ylim=state['score_bounds'],
            focus=state['focus'],
            title=plot_title()
        )
        state.update(ax1=ax1, ax2=ax2)
        entity_text.set_text(entity_summary())

    def redraw(_=None):
        redraw_plot()
        counter, image = word_data(word_cats())
        draw_word_panel(ax_words, counter, image, word_title())
        fig.canvas.draw_idle()

    def on_score_range(_val):
        redraw_plot()
        fig.canvas.draw_idle()

    def on_click(event):
        toolbar = getattr(fig.canvas, 'toolbar', None)
        # To ignore clicks while zooming/panning
        if toolbar is not None and getattr(toolbar, 'mode', '') != '':
            return
        if event.inaxes not in (state['ax1'], state['ax2']) or event.xdata is None:
            return
        sel = selected()
        index = int(round(event.xdata))
        if not 0 <= index < len(sel):
            return
        category = sel[index]
        state['focus'] = None if state['focus'] == category else category
        redraw()

    def build_checkboxes(selected_set):
        if state['check_ax'] is not None: state['check_ax'].remove()
        ax = fig.add_axes([0.80, 0.56, 0.18, 0.36])
        ax.set_title('Categores', fontsize=10)
        category = data['cats']
        if category:
            check = CheckButtons(
                ax,
                category, 
                [c in selected_set for c in category]
            )
            for label in check.labels:
                label.set_fontsize(7)
            check.on_clicked(redraw)
        else:
            ax.set_axs_off()
            ax.text(0.5, 0.5, 'No categories\nfor these settings', ha='center', va='center', fontsize=8)
            check = None
        state.update(check=check, check_ax=ax)

    def build_score_slider():
        old = state['score_slider']
        score_low, score_high = _score_bounds(full['cats'], full['scores'])
        init = (score_low, score_high)
        if old is not None:
            low, high = old.val
            low, high = max(low, score_low), min(high, score_high)
            if low < high:
                init = (low, high)
            old.ax.remove()
        ss = RangeSlider(fig.add_axes([0.10, 0.10, 0.37, 0.03]), 'Score range', score_low, score_high, valinit=init)
        ss.on_changed(on_score_range)
        state.update(score_slider=ss, score_bounds=(score_low, score_high))

    btn_all = Button(fig.add_axes([0.80, 0.50, 0.085, 0.04]), 'Select All')
    btn_none = Button(fig.add_axes([0.895, 0.50, 0.085, 0.04]), 'Select None')
    btn_apply = Button(fig.add_axes([0.80, 0.18, 0.18, 0.045]), 'Apply Changes')
    btn_save = Button(fig.add_axes([0.80, 0.125, 0.18, 0.045]), 'Save to PNG')

    tb_window = TextBox(fig.add_axes([0.89, 0.36, 0.09, 0.032]), 'Character Window ', initial=str(params['character_window']))
    tb_score = TextBox(fig.add_axes([0.89, 0.24, 0.09, 0.032]), 'Score Threshold ', initial='' if params['score'] is None else str(params['score']))

    def refresh_after_data_change(previous_selection, previous_category):
        keep = {c for c in data['cats'] if c in previous_selection or c not in previous_category}
        if data['cats'] != list(previous_cats_order[0]):
            build_checkboxes(keep)
        previous_cats_order[0] = list(data['cats'])
        redraw()

    previous_cats_order = [[]]

    def update_window_data():
        if not state['window_on']: return
        previous_selection, previous_category = set(selected()), set(data['cats'])
        try:
            load_view()
        except Exception as e:
            return
        refresh_after_data_change(previous_selection, previous_category)

    timer = fig.canvas.new_timer(interval=max(1, debounce_ms))
    timer.single_shot = True
    timer.add_callback(update_window_data)

    def set_all(target):
        check = state['check']
        if check is None: 
            return
        check.eventson = False
        for i, on in enumerate(check.get_status()):
            if on != target:
                check.set_active(i)
        check.eventson = True
        redraw()

    def apply(_):
        try:
            new_params = {
                'score': _parse_score(tb_score.text),
                'character_window': _parse_int(tb_window.text, 'Window'),
                'total_chars': 10_000
            }

            if new_params['character_window'] <= 0:
                raise ValueError('Character window must be greater than 0')
            if new_params['total_chars'] <= 0:
                raise ValueError('Total characters must be greater than 0')

        except ValueError as e:
            return

        fig.canvas.draw()
        fig.canvas.flush_events()

        previous_selection, previous_category = set(selected()), set(data['cats'])
        old_params = dict(params)
        old_full = dict(full)
        old_data = dict(data)

        window_changed = (new_params['character_window'] != params['character_window'] or 
                          new_params['total_chars'] != params['total_chars'])

        params.update(new_params)

        try:
            load_full()
            if window_changed: None
            load_view()
        except Exception as e:
            params.clear()
            params.update(old_params)
            full.clear()
            full.update(old_full)
            data.clear()
            data.update(old_data)
            return

        keep = {c for c in data['cats'] if c in previous_selection or c not in previous_category}
        build_checkboxes(keep)
        previous_cats_order[0] = list(data['cats'])
        build_score_slider()
        redraw()

    def save(_):
        out = Figure(figsize=(10, 6))
        ax = out.add_subplot()
        draw_category_plot(
            ax,
            selected(),
            data['counts'],
            data['scores'],
            score_range=score_range(),
            count_ylim=count_ylim(),
            score_ylim=state['score_bounds'],
            title=f'{plot_title()}\n{entity_summary()}'
        )
        out.tight_layout()
        out.savefig(save_path)

        counter, image = word_data(word_cats())
        wout = Figure(figsize=(7, 7))
        draw_word_panel(
            wout.add_subplot(),
            counter,
            image,
            f'{word_title(hint=False)} ({window_label()})'
        )
        wout.tight_layout()
        wout.savefig(wordcloud_path)

    btn_all.on_clicked(lambda _: set_all(True))
    btn_none.on_clicked(lambda _: set_all(False))
    btn_apply.on_clicked(apply)
    btn_save.on_clicked(save)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig._dashboard_widgets = (
        state,
        timer,
        btn_all,
        btn_none,
        btn_apply,
        btn_save,
        tb_window,
        tb_score
    )

    load_full()
    load_view()
    build_checkboxes(set(data['cats'] if initial is None else initial))
    previous_cats_order[0] = list(data['cats'])
    build_score_slider()
    redraw()
    return fig

if __name__ == '__main__':
    create_data_analysis_tool()
    plt.show()


