import json,sys
def blocks_chars(content):
    vis=0; red=0
    for b in content if isinstance(content,list) else [{'type':'text','text':content}]:
        t=b.get('type')
        if t=='text': vis+=len(b['text'])
        elif t=='tool_use': vis+=len(b['name'])+len(json.dumps(b['input']))
        elif t=='thinking': vis+=len(b.get('thinking',''))
        elif t=='redacted_thinking': red+=len(b['data'])
        elif t=='tool_result':
            c=b.get('content')
            if isinstance(c,list): vis+=sum(len(x.get('text','')) for x in c)
            else: vis+=len(str(c or ''))
        else: vis+=len(json.dumps(b))
    return vis,red
def timeline(path):
    rows=[json.loads(l) for l in open(path)]
    out=[]; cur=None
    for r in rows:
        if r.get('type')=='assistant':
            m=r['message']; v,rd=blocks_chars(m['content'])
            if cur and cur['id']==m['id']:
                cur['vis']+=v; cur['red']+=rd
            else:
                u=m['usage']
                cur={'kind':'response','id':m['id'],'vis':v,'red':rd,'in':u['input_tokens'],'cr':u.get('cache_read_input_tokens',0),'cc':u.get('cache_creation_input_tokens',0),'out':u['output_tokens'],'think':(u.get('output_tokens_details') or {}).get('thinking_tokens',0)}
                out.append(cur)
        elif r.get('type')=='user':
            v,_=blocks_chars(r['message']['content'])
            out.append({'kind':'input','vis':v}); cur=None
    return out
if __name__=='__main__':
    print(json.dumps(timeline(sys.argv[1])))
