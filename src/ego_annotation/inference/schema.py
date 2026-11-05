"""Strict sparse-window response validation; never repairs model responses."""
import json
CONTINUES={'none','from_previous','to_next','both'}
def validate(text,indices):
    x=json.loads(text); allowed=set(indices)
    if not isinstance(x,dict) or set(x)!={'segments','uncertainties'}: raise ValueError('Wrong top-level fields')
    if not isinstance(x['segments'],list) or not isinstance(x['uncertainties'],list): raise ValueError('Expected lists')
    for s in x['segments']:
        if not isinstance(s,dict) or set(s)!={'start_frame','end_frame','action','description','evidence_frames','continues'}: raise ValueError('Wrong segment fields')
        if type(s['start_frame']) is not int or type(s['end_frame']) is not int or s['start_frame'] not in allowed or s['end_frame'] not in allowed or s['start_frame']>s['end_frame']: raise ValueError('Invalid sparse bounds')
        if not isinstance(s['action'],str) or not s['action'] or not isinstance(s['description'],str) or not s['description']:raise ValueError('Expected action/description text')
        if s['continues'] not in CONTINUES: raise ValueError('Invalid continues')
        if not isinstance(s['evidence_frames'],list) or not s['evidence_frames'] or any(type(i) is not int or i not in allowed or not s['start_frame']<=i<=s['end_frame'] for i in s['evidence_frames']):raise ValueError('Invalid evidence frames')
    for u in x['uncertainties']:
        if not isinstance(u,dict) or set(u)!={'start_frame','end_frame','issue'}:raise ValueError('Wrong uncertainty fields')
        if type(u['start_frame']) is not int or type(u['end_frame']) is not int or u['start_frame'] not in allowed or u['end_frame'] not in allowed or u['start_frame']>u['end_frame'] or not isinstance(u['issue'],str) or not u['issue']:raise ValueError('Invalid uncertainty')
    return x

def prompt(indices):
    return '''Describe the visible clothing-handling actions in this one window, using only the ordered images below. Return ONLY one JSON object, without markdown or reasoning. Schema:
{"segments":[{"start_frame":integer,"end_frame":integer,"action":"short_english_snake_case","description":"One concrete English sentence about the visible action.","evidence_frames":[integer],"continues":"none|from_previous|to_next|both"}],"uncertainties":[{"start_frame":integer,"end_frame":integer,"issue":"short question or limitation"}]}
All frame numbers must be from these actual input frames: '''+json.dumps(indices)+'''. Segment bounds are INCLUSIVE coarse candidates anchored to shown frames, not precise action boundaries. evidence_frames must be actual shown frames inside that segment. Use a small number of segments that the images support. continues describes only what is visible at this window's edges: from_previous means already underway at the first shown frame, to_next means still underway at the last shown frame, both means both edges; none means neither. These flags do not assert anything about unseen neighboring windows. If uncertain, describe the visible motion conservatively and add a bounded uncertainty; use [] if none. Do not add confidence, observations, metadata, or an overall task-success claim. Do not invent actions between images. The images are ordered by acquisition time.'''

def parse_response(text,indices):
    """Remove only a single complete outer Markdown wrapper; never repair JSON content."""
    import re
    strict=False;valid=False;removed=False;body=text;parsed=None;error=None
    try:json.loads(text);strict=True;valid=True
    except (ValueError,TypeError):
        match=re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*',text,flags=re.S)
        if match:
            body=match.group(1)
            try:json.loads(body);valid=True;removed=True
            except (ValueError,TypeError):pass
    try:parsed=validate(body,indices)
    except Exception as e:error=str(e)
    return parsed,error,valid,strict,removed
