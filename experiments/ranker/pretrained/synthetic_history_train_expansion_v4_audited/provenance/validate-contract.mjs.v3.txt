// CPU public DTO validation only; no model, browser or private annotation imports.
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const here=dirname(fileURLToPath(import.meta.url)),root=resolve(here,'../../../..');
const require=createRequire(resolve(root,'package.json'));
const ts=require('tsx/cjs/api').require;
const {validatePublicHistoryRequest}=ts('./packages/contracts/src/synthetic-history-v1.ts',resolve(root,'package.json'));
const {profiles}=JSON.parse(await readFile(resolve(here,'profiles.train.public.json'),'utf8'));
for(const profile of profiles)validatePublicHistoryRequest(profile.request);
console.log(JSON.stringify({status:'public-contract-pass',validated:profiles.length,privateAnnotationsRead:false,modelOrBrowserLoaded:false}));
