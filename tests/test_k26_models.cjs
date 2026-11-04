'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const repo = path.resolve(__dirname, '..');
const project = path.dirname(repo);
const csvLines=fs.readFileSync(path.join(project,'data/processed/params_groups.csv'),'utf8').trim().split(/\r?\n/);
const csvColumns=csvLines.shift().split(',');
const parameters=Object.fromEntries(csvLines.map(line=>{const row=Object.fromEntries(line.split(',').map((value,i)=>[csvColumns[i],value]));return [row.group,row];}));
const files = {
  'simulator-en.html': path.join(project, 'outputs/dashboard/index_v3.html'),
  'simulator-he.html': path.join(repo, 'voting-simulator/index.html'),
  'legacy.html': path.join(project, 'external_legacy_release_repo/index.html'),
  'propensity.html': path.join(repo, 'voting-simulator/demographics.html'),
  'demographics.html': path.join(repo, 'voting-simulator/demographics.html'),
};
(async () => {
  const browser = await chromium.launch({headless:true, ...(process.env.BROWSER_EXECUTABLE ? {executablePath:process.env.BROWSER_EXECUTABLE} : {})});
  const errors=[];
  try {
    const page=await browser.newPage({viewport:{width:1280,height:960}});
    page.on('pageerror', error=>errors.push(error.message));
    await page.route('**/*', route=>{
      const url=new URL(route.request().url());
      const file=files[url.pathname.slice(1)];
      if(url.hostname==='model.test' && file) return route.fulfill({contentType:'text/html',body:fs.readFileSync(file)});
      return route.abort();
    });
    for(const name of ['simulator-en.html','simulator-he.html','legacy.html']) {
      await page.goto('https://model.test/'+name+'#2026');
      assert.ok(await page.locator('#tab-k26').evaluate(el=>el.classList.contains('active')),'Comparison deep link opens its tab');
      assert.equal(await page.locator('.tabs-bar .tab-btn').last().getAttribute('data-tab'),'k26','Comparison is the final tab');
      await page.locator('[data-tab="k26"]').click();
      assert.equal(await page.locator('#k26Comparison tbody tr').count(),6);
      assert.equal(await page.locator('#k26Comparison p').count(),0,'Comparison contains no narrative paragraphs');
      assert.ok((await page.locator('#k26Comparison th').nth(1).textContent()).includes('2022'),'Population-change header states its true reference');
      const cells=await page.locator('#k26Comparison tbody td').allTextContents();
      assert.ok(cells.every(cell=>!cell.includes('—')),'No inapplicable SD placeholders in the comparison');
      assert.ok(cells.every(cell=>!/[0-9]+\.[0-9]+%/.test(cell)),'Percentages use whole numbers');
      assert.ok(cells.filter(cell=>cell.includes(' SD')).every(cell=>/[+-][0-9]+\.[0-9] SD/.test(cell)),'SD changes use one decimal');
      assert.equal((await page.locator('#k26Comparison tbody tr').first().locator('td').allTextContents())[1],'11% (+0.6 pp)');
      const before=await page.evaluate(()=>({...currentVal}));
      for(const [group,row] of Object.entries(parameters)) {
        assert.ok(Math.abs(before['t_'+group]-Number(row.theta_vote_mean))<1e-12,'Published turnout default must match model output: '+group);
        assert.ok(Math.abs(before['k_'+group]-Number(row.theta_bibi_mean))<1e-12,'Published support default must match model output: '+group);
      }
      await page.locator('#loadK26').click();
      assert.ok(await page.locator('#tab-scenario').evaluate(el=>el.classList.contains('active')),'Loading estimates opens the scenario');
      assert.equal(await page.locator('.load-btn.active').getAttribute('data-election'),'K26');
      assert.equal(await page.locator('#wc').inputValue(),'2.34');
      assert.equal(await page.locator('#wo').inputValue(),'3.04');
      const actual=await page.evaluate(()=>({...currentVal}));
      const expected=JSON.parse(fs.readFileSync(path.join(project,'data/processed/k26/current/model_comparison/six_group_comparison.json'),'utf8'));
      for(const row of expected.rows) {
        const key=(row.parameter==='theta_vote'?'t':'k')+'_'+row.group;
        assert.ok(Math.abs(actual[key]-row.k26_estimate)<1e-8,key);
      }
      const metadata=await page.locator('#k26ModelMetadata').textContent().then(JSON.parse);
      assert.equal(metadata.projection_years,3.5);
      assert.ok(Math.abs(metadata.coverage.k26_matched_valid_vote_fraction_national-.9131946897)<1e-8);
      await page.locator('#clearBtn').click();
      assert.deepEqual(await page.evaluate(()=>({...currentVal})),before,'Reset must restore frozen defaults');
      assert.equal(await page.locator('.sd-btn[data-n="3"]').count(),12,'Every control includes +3 SD');
      await page.locator('#t_masorti_btns [data-n="3"]').click();
      assert.ok(Math.abs(await page.evaluate(()=>currentVal.t_masorti)-(Number(parameters.masorti.theta_vote_mean)+3*Number(parameters.masorti.theta_vote_sd)))<1e-12);
      assert.equal(await page.locator('#t_masorti_sd').textContent(),'+3.0 SD');
      await page.locator('#k_dati_btns [data-n="3"]').click();
      assert.equal(await page.evaluate(()=>currentVal.k_dati),1,'Probability remains bounded at 100%');
      assert.equal((await page.locator('#k_dati_sd').textContent()).trim(),'+1.0 SD','Readout reports achieved SD after the probability cap');
      assert.ok(await page.locator('#k_haredi_btns [data-n="3"]').isDisabled());
      assert.equal(await page.locator('#k_haredi_sd').textContent(),'','Inapplicable SD is omitted');
      assert.ok(await page.locator('#k_haredi_sd').isHidden());
      await page.locator('#clearBtn').click();
      await page.locator('.load-btn[data-election="K26"]').click();
      assert.deepEqual(await page.evaluate(()=>({...currentVal})),actual,'Normal 2026 button loads the same election estimates');
      assert.equal(await page.locator('.load-btn.active').getAttribute('data-election'),'K26');
      assert.equal((await page.locator('#t_masorti_sd').textContent()).trim(),'+3.7 SD','Election loading shows its actual historical SD movement');
      await page.locator('.load-btn[data-election="K25"]').click();
      assert.equal(await page.locator('.load-btn.active').getAttribute('data-election'),'K25');
      for(const [group,row] of Object.entries(parameters)) {
        const loaded=await page.evaluate(group=>({t:currentVal['t_'+group],k:currentVal['k_'+group]}),group);
        assert.ok(Math.abs(loaded.t-Number(row.theta_vote_K25))<1e-12,'2022 turnout still loads');
        assert.ok(Math.abs(loaded.k-Number(row.theta_bibi_K25))<1e-12,'2022 support still loads');
      }
      await page.locator('#clearBtn').click();
      await page.locator('[data-tab="parties"]').click();
      assert.equal(await page.locator('#partyElection').inputValue(),'K26','Party tagging opens on the latest election');
      assert.equal(await page.locator('#partiesTable tbody tr').count(),38,'2026 party tagging preserves every list');
      const partyData=await page.evaluate(()=>({sum:DATA.parties.K26.reduce((sum,p)=>sum+p.share,0),amcha:DATA.parties.K26.find(p=>p.letter==='ך').bloc,added:['ני','נר','זך','בי'].map(symbol=>DATA.parties.K26.find(p=>p.letter===symbol).bloc),coalitionShare:DATA.parties.K26.filter(p=>p.bloc==='Coalition').reduce((sum,p)=>sum+p.share,0)}));
      assert.ok(Math.abs(partyData.sum-1)<1e-12,'Displayed K26 lists exhaust national valid votes');
      assert.equal(partyData.amcha,'Coalition');
      assert.deepEqual(partyData.added,['Coalition','Coalition','Coalition','Coalition'],'All four reviewed minor lists count as coalition');
      assert.ok(Math.abs(partyData.coalitionShare-expected.national_observed.coalition_share_valid)<1e-12,'Party tagging and fitted bloc use the same definition');
      await page.selectOption('#partyElection','K25');
      assert.ok(await page.locator('#partiesTable tbody tr').count()>0,'Historical party tagging still works');
      await page.selectOption('#partyElection','K26');
      await page.locator('[data-tab="k26"]').click();
      await page.setViewportSize({width:390,height:844});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+2),'Mobile page must not overflow');
      await page.setViewportSize({width:1280,height:960});
    }
    await page.goto('https://model.test/propensity.html');
    assert.equal(await page.locator('#modelElection').inputValue(),'historical');
    const frozen=await page.evaluate(()=>({columns:CUBE.columns,rows:CUBE.rows}));
    const baseline=JSON.stringify(frozen.rows);
    assert.equal(await page.locator('#k26ModelPanel p').count(),0,'No methodological narrative in the demographic controls');
    assert.equal(await page.locator('#k26ModelStatus, #k26FitCoverage').count(),0,'Technical audit text is absent from viewer flow');
    await page.locator('#k26ModelPanel details').evaluate(el=>el.open=true);
    const modelFolder=path.join(project,'data/processed/k26/current/individual_model');
    const readJson=file=>JSON.parse(fs.readFileSync(file,'utf8'));
    const elections=['K21','K22','K23','K24','K25','K26'];
    const history=readJson(path.join(modelFolder,'historical_election_cubes.json'));
    const latest=readJson(path.join(modelFolder,'k26_probability_cube.json'));
    const jointSources={...history,K26:latest};
    const families=await page.evaluate(()=>P.map(p=>({key:p[0],label:p[2]})));
    const columns=Object.fromEntries(frozen.columns.map((name,index)=>[name,index]));
    const cellKeys=['pop_group','religiosity','income_band','age_band','academic','district'];
    function mergeSource(cube){
      const source=Object.fromEntries(cube.columns.map((name,index)=>[name,index]));
      const cellKey=(row,index)=>cellKeys.map(key=>row[index[key]]).join('|');
      const byCell=new Map(cube.rows.map(row=>[cellKey(row,source),row]));
      assert.equal(byCell.size,frozen.rows.length,'Source fits cover every frozen demographic cell');
      return frozen.rows.map(row=>{
        const fitted=byCell.get(cellKey(row,columns));
        assert.ok(fitted,'Source fit has matching demographic identities');
        return row.map((value,index)=>{
          const name=frozen.columns[index];
          return name.endsWith('_mass')?fitted[source[name]]:value;
        });
      });
    }
    function values(rows,selection){
      const included=rows.filter(row=>Object.entries(selection).every(([name,allowed])=>allowed.includes(String(row[columns[name]]))));
      const sum=name=>included.reduce((total,row)=>total+row[columns[name]],0);
      const weight=sum('weight'),voted=weight-sum('abstain_mass');
      return [weight?100*voted/weight:null,...families.map(party=>voted?100*sum(party.key+'_mass')/voted:null)];
    }
    const percentage=value=>value===null?'':value.toFixed(0)+'%';
    const signed=value=>value===null?'':(value>=0?'+':'')+value.toFixed(1);
    const activeSelection=()=>page.evaluate(()=>Object.fromEntries(Object.entries(F).map(([key,el])=>[key,[...el.querySelectorAll('input:checked')].map(input=>input.value)])));
    async function checkComparison(cubes,a,b){
      const selected=await activeSelection();
      const frozenValues=values(frozen.rows,selected);
      const byElection=Object.fromEntries(Object.entries(cubes).map(([election,rows])=>[election,values(rows,selected)]));
      const historical=elections.slice(0,5).map(election=>byElection[election]);
      const mean=frozenValues.map((_,i)=>historical.every(row=>row[i]!==null)?historical.reduce((sum,row)=>sum+row[i],0)/5:null);
      const first=a==='mean'?mean:byElection[a],second=b==='mean'?mean:byElection[b];
      const expected=frozenValues.map((old,i)=>{
        const sd=mean[i]===null?null:Math.sqrt(historical.reduce((sum,row)=>sum+(row[i]-mean[i])**2,0)/5);
        const delta=first[i]===null||second[i]===null?null:second[i]-first[i];
        const z=delta===null||sd===null||sd<.0001?null:delta/sd;
        return [percentage(old),percentage(first[i]),percentage(second[i]),signed(delta),signed(z)];
      });
      const displayed=await page.locator('#k26FamilyComparison tr').evaluateAll(rows=>rows.map(row=>[...row.querySelectorAll('td')].slice(1).map(cell=>cell.textContent.trim())));
      assert.deepEqual(displayed,expected,'Filtered comparison uses selected elections and historical population SD: '+a+' to '+b);
      assert.equal(await page.locator('#k26FamilyComparison tr').first().locator('td').first().textContent(),'שיעור הצבעה');
      for(const side of ['A','B']) {
        const selectedLabel=await page.locator('#comparisonElection'+side).evaluate(select=>select.selectedOptions[0].textContent);
        assert.equal(await page.locator('#comparisonHeader'+side).textContent(),selectedLabel,'Comparison header identifies its selected election');
      }
    }
    // The page must expose each election as a peer; these checks also detect a selector that only relabels the current cube.
    const fitOptions=[...elections,'joint_combined'];
    assert.deepEqual(await page.locator('#modelElection option').evaluateAll(options=>options.map(option=>option.value)),['historical',...fitOptions]);
    assert.equal(await page.locator('#modelFitMethod').count(),0,'All annual and combined fits share one selector');
    for(const id of ['comparisonElectionA','comparisonElectionB']) {
      assert.deepEqual(await page.locator('#'+id+' option').evaluateAll(options=>options.map(option=>option.value)),['mean',...fitOptions]);
    }
    const jointCubes=Object.fromEntries(elections.map(election=>[election,mergeSource(jointSources[election])]));
    const independentFolder=path.join(modelFolder,'independent');
    const independentAudit=readJson(path.join(independentFolder,'model_audit.json'));
    assert.equal(independentAudit.publication_approved,true,'Annual fits require an approved independent bundle');
    const independent=readJson(path.join(independentFolder,'election_cubes.json'));
    assert.deepEqual(Object.keys(independent).sort(),elections,'Independent bundle contains all six elections');
    const cubes=Object.fromEntries(elections.map(election=>[election,mergeSource(independent[election])]));
    const provenance=await page.evaluate(()=>K26_FIT_METADATA.methods.independent);
    assert.equal(provenance.collection_id,independentAudit.collection_id,'Published independent provenance matches the approved audit');
    for(const election of elections) {
      assert.equal(independent[election].election,election,'Independent cube carries its own election identity');
      assert.equal(provenance.election_fit_ids[election],independent[election].model_fit_id,'Published fit identity matches '+election+' source');
    }
    // Average joint probability masses per demographic cell, then calculate conditional vote shares.
    // Averaging six already-conditioned party percentages would give a different result.
    cubes.joint_combined=frozen.rows.map((row,index)=>row.map((value,column)=>frozen.columns[column].endsWith('_mass')?elections.reduce((sum,election)=>sum+jointCubes[election][index][column],0)/6:value));
    for(const election of fitOptions) {
      await page.selectOption('#modelElection',election);
      const displayedRows=await page.evaluate(()=>CUBE.rows);
      if(election==='joint_combined') {
        assert.equal(displayedRows.length,cubes[election].length);
        displayedRows.forEach((row,index)=>row.forEach((value,column)=>{
          const expected=cubes[election][index][column];
          if(frozen.columns[column].endsWith('_mass')) {
            // Python's compensated float sum can differ from JavaScript's reduction in the final bits.
            assert.ok(Math.abs(value-expected)<=1e-10*Math.max(1,Math.abs(expected)),'Combined cell is the equal six-election mass mean');
          } else assert.equal(value,expected,'Combined cube preserves frozen demographic/fiscal cells');
        }));
      } else assert.deepEqual(displayedRows,cubes[election],election+' loads its exact independent source masses and preserves frozen population/fiscal inputs');
      const actualLabels=await page.locator('#partyA input').evaluateAll(inputs=>inputs.map(input=>({key:input.value,label:input.dataset.label})));
      assert.deepEqual(actualLabels,families,'Party controls use stable political-family labels in '+election);
      assert.deepEqual(await page.evaluate(()=>P.map(p=>({key:p[0],label:p[2]}))),families,'Family identities remain stable across elections');
      assert.equal(await page.locator('#labelA').textContent(),families.find(party=>party.key==='likud').label,'Party comparison uses the stable selected family name');
      assert.equal(await page.locator('#labelB').textContent(),families.find(party=>party.key==='center').label,'Historical center labels do not inherit 2026 party aliases');
      await page.selectOption('#comparisonElectionA',election);
      await page.selectOption('#comparisonElectionB','K26');
      await checkComparison(cubes,election,'K26');
    }
    await page.selectOption('#comparisonElectionA','mean');
    await page.selectOption('#comparisonElectionB','K26');
    await checkComparison(cubes,'mean','K26');
    await page.selectOption('#comparisonElectionA','joint_combined');
    await checkComparison(cubes,'joint_combined','K26');
    await page.selectOption('#comparisonElectionA','K25');
    await page.selectOption('#comparisonElectionB','joint_combined');
    await checkComparison(cubes,'K25','joint_combined');
    const combinedValues=values(cubes.joint_combined,await activeSelection());
    const jointValueRows=elections.map(election=>values(jointCubes[election],{}));
    assert.ok(combinedValues.slice(1).some((value,index)=>Math.abs(value-jointValueRows.reduce((sum,row)=>sum+row[index+1],0)/6)>1e-6),'Data distinguishes mass-derived conditional shares from an average of six party percentages');
    assert.equal(await page.locator('#turnout').textContent(),combinedValues[0].toFixed(1)+'%','Combined turnout derives from averaged cell masses');
    const unfiltered=await page.locator('#k26FamilyComparison').textContent();
    await page.evaluate(()=>{
      const inputs=[...F.pop_group.querySelectorAll('input')];
      inputs.forEach((input,index)=>input.checked=index===0);
      inputs[0].dispatchEvent(new Event('change',{bubbles:true}));
    });
    await checkComparison(cubes,'K25','joint_combined');
    assert.notEqual(await page.locator('#k26FamilyComparison').textContent(),unfiltered,'Comparison follows demographic filters');
    await page.selectOption('#comparisonElectionA','joint_combined');
    await page.selectOption('#comparisonElectionB','K26');
    await checkComparison(cubes,'joint_combined','K26');
    await page.selectOption('#comparisonElectionA','K25');
    await page.selectOption('#comparisonElectionB','K21');
    await checkComparison(cubes,'K25','K21');
    await page.locator('#filters .trigger').first().click();
    await page.locator('#filters [data-x="none"]').first().click();
    await checkComparison(cubes,'K25','K21');
    assert.ok(await page.locator('#k26FamilyComparison tr').evaluateAll(rows=>rows.every(row=>[...row.querySelectorAll('td')].slice(1).every(cell=>cell.textContent===''))),'Empty group has blank numeric cells in every row');
    await page.locator('#filters [data-x="all"]').first().click();
    await page.selectOption('#modelElection','historical');
    assert.equal(await page.evaluate(()=>JSON.stringify(CUBE.rows)),baseline,'Frozen published cube restores exactly after independent and combined selections');
    await page.locator('body').click({position:{x:5,y:5}});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+2),'Desktop propensity explorer must not overflow');
    await page.setViewportSize({width:390,height:844});
    await page.selectOption('#modelElection','K26');
    await page.locator('#k26ModelPanel details').evaluate(el=>el.open=true);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+2),'Mobile propensity explorer must not overflow');
    await page.selectOption('#modelElection','joint_combined');
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+2),'Mobile combined-fit selector must not overflow');
    await page.locator('#partyA .trigger').click();
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+2),'Mobile party controls must not overflow');
    assert.deepEqual(errors,[],'No browser exceptions');
    console.log('K26 browser checks passed: unchanged simulator coverage; six independent annual cubes, combined joint cell means, comparison/SD math, filters, family labels, frozen restore, desktop/mobile layout.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
