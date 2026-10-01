const fs=require('fs');
const html=fs.readFileSync(__dirname + '/index.html','utf8');
const js=html.split('<script>')[1].split('</script>')[0];
const el=()=>({innerHTML:'',addEventListener(){},classList:{toggle(){}},scrollTop:0,scrollHeight:0});
global.document={getElementById:el,querySelectorAll:()=>[]};
const run=new Function(js+`
return SCENARIOS.map((s,i)=>{current=i;reset();runUntil(()=>false);const c=state.reconciled;return [s.name,c.mode,fmt(c.estimate),fmt(c.actual),pct(c.error),c.darkEstimate>0?pct(c.darkError):'-',state.crossed.soft&&state.crossed.soft.responses,truthCross().replace('<br>',''),state.log.filter(l=>l.level!=='INFO').map(l=>l.event).join(',')].join(' | ')});`);
console.log(run().join('\n'));
