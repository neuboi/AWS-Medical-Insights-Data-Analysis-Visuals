# AWS Medical Insights Data Analysis Visuals

A collection of interactive data analysis visuals for AWS Medical Insights.

*To get started, copy your AWS Medical Insights transcripts (JSON) into the folder `json_files`*

## main.py
An intereactive plot of Medical Insight entity category frequency and confidence score distributions across all medical insights transcripts in `json_files`. Includes a word cloud of the text in selected categories

**How to use**:
* Check boxes (right) to choose which categories should be shown.
* Word cloud (middle) shows words of all ticked categories combined.
    * Click a bar/box to show only that category's words; Click the bar/box again to go back
* Range slider (bottom) to set score (y-axis) range
* Score box, Character window box, and "Apply All Changes" button re-runs create_variables() with new values and re-renders visual
* "All" / "None" toggles every category
* "Save PNG exports plot and word cloud

**Requirements**: 

    pip install wordcloud, matplotlib

**Usage**:

    python main.py
    
    OR
    
    from main import create_data_analysis_tool
    create_data_analysis_tool()
    plt.show()

## table.py
An intereactive table of n most frequent words in each category across all medical insights transcripts in `json_files`. Uses `google-10000-english-usa.txt` to filter out less specific terms.

**How to use**:
* Check boxes (left) to choose which categories should be shown.
* Slider (bottom) to set n most frequent words to display

**Requirements**: 

    pip install matplotlib

**Usage**:

    python table.py