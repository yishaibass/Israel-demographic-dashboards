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
      assert.equal(await page.locator('#wc').inputValue(),'1.68');
      assert.equal(await page.locator('#wo').inputValue(),'3.70');
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
      const partyData=await page.evaluate(()=>({sum:DATA.parties.K26.reduce((sum,p)=>sum+p.share,0),amcha:DATA.parties.K26.find(p=>p.letter==='ך').bloc}));
      assert.ok(Math.abs(partyData.sum-1)<1e-12,'Displayed K26 lists exhaust national valid votes');
      assert.equal(partyData.amcha,'Coalition');
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
    const baseline=await page.evaluate(()=>JSON.stringify(CUBE.rows));
    const before=await page.locator('#k26FamilyComparison').textContent();
    await page.selectOption('#modelElection','K26');
    assert.equal(await page.locator('#k26ModelPanel p').count(),0,'No methodological narrative in the demographic controls');
    assert.equal(await page.locator('#k26ModelStatus, #k26FitCoverage').count(),0,'Technical audit text is absent from viewer flow');
    const modelFolder=path.join(project,'data/processed/k26/current/individual_model');
    const history=JSON.parse(fs.readFileSync(path.join(modelFolder,'historical_election_cubes.json'),'utf8'));
    const latest=JSON.parse(fs.readFileSync(path.join(modelFolder,'k26_probability_cube.json'),'utf8'));
    function nationalTurnout(cube){
      const weight=cube.columns.indexOf('weight'),abstain=cube.columns.indexOf('abstain_mass');
      return 100*(1-cube.rows.reduce((sum,row)=>sum+row[abstain],0)/cube.rows.reduce((sum,row)=>sum+row[weight],0));
    }
    const historicalRates=Object.values(history).map(nationalTurnout);
    const expectedMean=historicalRates.reduce((sum,x)=>sum+x,0)/5;
    const expectedSd=Math.sqrt(historicalRates.reduce((sum,x)=>sum+(x-expectedMean)**2,0)/5);
    const cells=await page.locator('#k26FamilyComparison tr').first().locator('td').allTextContents();
    assert.equal(cells[2],expectedMean.toFixed(0)+'%','Common historical mean is calculated from five election rates');
    assert.equal(Number(cells[5]),Number(((nationalTurnout(latest)-expectedMean)/expectedSd).toFixed(1)),'SD change uses the same-design historical mean and population SD');
    assert.notEqual(await page.evaluate(()=>JSON.stringify(CUBE.rows)),baseline);
    assert.ok(await page.locator('#partyA').textContent().then(s=>s.includes('ישר וביחד')),'K26 party selectors use reviewed family names');
    await page.evaluate(()=>{
      const input=F.pop_group.querySelector('input[value="1"]');
      input.checked=false;input.dispatchEvent(new Event('change',{bubbles:true}));
    });
    assert.notEqual(await page.locator('#k26FamilyComparison').textContent(),before,'Comparison follows demographic filters');
    await page.locator('#filters .trigger').first().click();
    await page.locator('#filters [data-x="none"]').first().click();
    assert.ok(await page.locator('#k26FamilyComparison tr').first().locator('td').allTextContents().then(cells=>cells.slice(1).every(value=>value==='')),'Empty group has blank numeric cells');
    await page.selectOption('#modelElection','historical');
    assert.equal(await page.evaluate(()=>JSON.stringify(CUBE.rows)),baseline,'Historical cube restores exactly');
    await page.setViewportSize({width:390,height:844});
    await page.locator('#k26ModelPanel details').evaluate(el=>el.open=true);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+2),'Mobile propensity explorer must not overflow');
    assert.deepEqual(errors,[],'No browser exceptions');
    console.log('K26 browser checks passed: English/Hebrew simulators, apply/reset, frozen cube, filters, family labels, empty selection, mobile layout.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
