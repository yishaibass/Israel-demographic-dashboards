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
      await page.goto('https://model.test/'+name);
      assert.equal(await page.locator('#k26Comparison tbody tr').count(),12);
      const before=await page.evaluate(()=>({...currentVal}));
      for(const [group,row] of Object.entries(parameters)) {
        assert.ok(Math.abs(before['t_'+group]-Number(row.theta_vote_mean))<1e-12,'Published turnout default must match model output: '+group);
        assert.ok(Math.abs(before['k_'+group]-Number(row.theta_bibi_mean))<1e-12,'Published support default must match model output: '+group);
      }
      await page.locator('#loadK26').click();
      const actual=await page.evaluate(()=>({...currentVal}));
      const expected=JSON.parse(fs.readFileSync(path.join(project,'data/processed/k26/current/model_comparison/six_group_comparison.json'),'utf8'));
      for(const row of expected.rows) {
        const key=(row.parameter==='theta_vote'?'t':'k')+'_'+row.group;
        assert.ok(Math.abs(actual[key]-row.k26_estimate)<1e-8,key);
      }
      assert.ok((await page.locator('#k26Comparison').textContent()).includes('91.32%'));
      await page.locator('#clearBtn').click();
      assert.deepEqual(await page.evaluate(()=>({...currentVal})),before,'Reset must restore frozen defaults');
      await page.setViewportSize({width:390,height:844});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+2),'Mobile page must not overflow');
      await page.setViewportSize({width:1280,height:960});
    }
    await page.goto('https://model.test/propensity.html');
    assert.equal(await page.locator('#modelElection').inputValue(),'historical');
    const baseline=await page.evaluate(()=>JSON.stringify(CUBE.rows));
    const before=await page.locator('#k26FamilyComparison').textContent();
    await page.selectOption('#modelElection','K26');
    assert.ok(await page.locator('.top p').textContent().then(s=>s.includes('2026')),'Introduction identifies the selected election');
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
    assert.equal(cells[2],expectedMean.toFixed(2)+'%','Common historical mean is calculated from five election rates');
    assert.equal(Number(cells[5]),Number(((nationalTurnout(latest)-expectedMean)/expectedSd).toFixed(2)),'SD change uses the same-design historical mean and population SD');
    assert.notEqual(await page.evaluate(()=>JSON.stringify(CUBE.rows)),baseline);
    assert.ok(await page.locator('#k26FitCoverage').textContent().then(s=>s.includes('81.71%')));
    assert.ok(await page.locator('#partyA').textContent().then(s=>s.includes('ישר וביחד')),'K26 party selectors use reviewed family names');
    await page.evaluate(()=>{
      const input=F.pop_group.querySelector('input[value="1"]');
      input.checked=false;input.dispatchEvent(new Event('change',{bubbles:true}));
    });
    assert.notEqual(await page.locator('#k26FamilyComparison').textContent(),before,'Comparison follows demographic filters');
    await page.locator('#filters .trigger').first().click();
    await page.locator('#filters [data-x="none"]').first().click();
    assert.ok(await page.locator('#k26FamilyComparison').textContent().then(s=>s.includes('—')),'Empty group remains missing');
    await page.selectOption('#modelElection','historical');
    assert.equal(await page.evaluate(()=>JSON.stringify(CUBE.rows)),baseline,'Historical cube restores exactly');
    await page.setViewportSize({width:390,height:844});
    await page.locator('#k26ModelPanel details').evaluate(el=>el.open=true);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+2),'Mobile propensity explorer must not overflow');
    assert.deepEqual(errors,[],'No browser exceptions');
    console.log('K26 browser checks passed: English/Hebrew simulators, apply/reset, frozen cube, filters, family labels, empty selection, mobile layout.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
