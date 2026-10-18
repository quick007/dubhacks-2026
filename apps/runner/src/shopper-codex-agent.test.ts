import assert from "node:assert/strict";
import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { buildShopperAssignment } from "./shopper-contract.js";
import { loadShopperPacket } from "./shopper-eval-cli.js";
import { CodexShopperAgent, codexModelAvailability, findCodexBinary } from "./shopper-codex-agent.js";

test("official CLI adapter protocol fixture verifies models and isolated structured screenshot decisions", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "codex-adapter-fixture-"));
  const binary = path.join(root, "fixture-codex");
  try {
    await writeFile(binary, `#!/usr/bin/env node
const fs=require('node:fs'), readline=require('node:readline');
if(process.argv.includes('app-server')) {
 readline.createInterface({input:process.stdin}).on('line',l=>{const p=JSON.parse(l);
  if(p.method==='initialize')console.log(JSON.stringify({id:p.id,result:{}}));
  if(p.method==='model/list')console.log(JSON.stringify({id:p.id,result:{data:[{model:'gpt-6.1-sol',inputModalities:['text','image']}]}}));
 });
}else{
 let input='';process.stdin.on('data',b=>input+=b);process.stdin.on('end',()=>{
  const p=JSON.parse(input),a=process.argv;
  if(!a.includes('--ignore-user-config')||!a.includes('--ephemeral')||!a.includes('read-only')||!a.includes('--no-daemon'))process.exit(12);
  if(p.assignment.gold||p.assignment.catalog||fs.readdirSync(process.cwd()).some(n=>n.includes('gold')))process.exit(13);
  if(!fs.existsSync(a[a.indexOf('--image')+1]))process.exit(14);
  console.log(JSON.stringify({type:'item.completed',item:{type:'agent_message',text:JSON.stringify({action:{type:'search',query:p.assignment.recommenderContext.query}})}}));
 });
}
`);
    await chmod(binary, 0o755);
    assert.equal(await findCodexBinary(binary), binary);
    assert.equal((await codexModelAvailability(binary, "gpt-6.1-sol")).model, "gpt-6.1-sol");
    await assert.rejects(codexModelAvailability(binary, "invented-model"), /No model substituted/);
    const packet = await loadShopperPacket(path.resolve("experiments/ranker/pretrained/store_shopper_eval_v1"));
    const assignment = buildShopperAssignment(packet.cases[1], "generic");
    const log: unknown[] = [], agent = new CodexShopperAgent(binary, "gpt-6.1-sol", assignment, (event) => log.push(event));
    try {
      const action = await agent.decide({ text: "Fixture only", url: assignment.entryUrl, controls: [], screenshotBase64: Buffer.from("fixture image").toString("base64") }, { actions: 24, wallMs: 1000 });
      assert.deepEqual(action, { type: "search", query: assignment.recommenderContext.query });
      assert.equal(log.length, 2);
    } finally { await agent.close(); }
  } finally { await rm(root, { recursive: true, force: true }); }
});
