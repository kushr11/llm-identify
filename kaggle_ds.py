import pandas as pd
import re

def load_file(path, names):
    with open(path, 'r') as f:
        lines = f.readlines() 
    return pd.DataFrame(lines, columns=names)

def load_data(basePath):
  tags = {'WP':'Writing Prompt',
  'SP':'Simple Prompt',
  'EU':'Established Universe',
  'CW':'Constrained Writing',
  'TT':'Theme Thursday',
  'PM':'Prompt Me',
  'MP':'Media Prompt',
  'IP':'Image Prompt',
  'PI':'Prompt Inspired',
  'OT':'Off Topic',
  'RF':'Reality Fiction'}

  dfConcat = pd.DataFrame()
  for split in ['train', 'valid', 'test']:
    df = load_file(f'{basePath}/writingPrompts/{split}.wp_source', ['prompt'])
    for tag in tags.keys():
      df[tag.lower()] = df['prompt'].map(lambda x: check_tag(x, tag.lower()))
    df['tagCounter']= df.iloc[:,[2,-1]].sum(axis=1)
    df['splitLineIndex'] = df.index
    story = load_file(f'{basePath}/writingPrompts/{split}.wp_target', ['story'])
    df['story'] = story['story']
    df['split'] = split
    dfConcat = pd.concat([dfConcat, df])
  return dfConcat

def check_tag(item, tag):
  r=re.compile(r'[\(\{\[]\s*[\w]{2}\s*[\]\}\)]\s*')
  m=r.findall(item.lower())
  if len(m) > 0:
    for group in m:
      if tag in group:
        return 1
  return 0

def show_data(df):
    html_string = '''
                <html>
                  <head><title>HTML Pandas Dataframe with CSS</title></head>
                  <link rel="stylesheet" type="text/css" href="df_style.css"/>
                  <body>
                    {table}
                  </body>
                </html>.
                '''
    df = df.replace('\<newline\>|\< newline \>|\<new line\>', '\n', regex=True)
    df.style.set_properties(**{'text-align': 'left'}).set_table_styles([ dict(selector='th', props=[('text-align', 'left')] ) ])
    html = df.to_html()
    html_string = html_string.format(table=html)
    html_string = html_string.replace(r'\n','<br>' ).\
                              replace('<td>', '<td style="text-align:left">').\
                              replace('<th>', '<th style="text-align:left">')
    display(HTML(html_string))  

def get_samples(df, n, constraint = None, show = True):
    samples = zip(df['prompt'].iloc[:n,0].index, df['prompt'].iloc[:n,0], df['story'].iloc[:n,0])
    df = pd.DataFrame(samples, columns=['index', 'prompt', 'story'])
    if constraint is not None:
        df = df[df['prompt'].str.contains(constraint)]
    return df



#df = load_data('/data1/ybaiaj/llm-identify-main-new-6.7b-25/writing-prompts')
#df_iterator = df.iterrows()
#_,row_data = next(df_iterator)
#col_val = row_data['story']
#print(col_val[:10])