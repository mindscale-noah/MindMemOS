const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
function loadSource(name, dependencies = {}) {
  const source = fs.readFileSync(path.join(__dirname, '../src', name + '.ts'), 'utf8');
  const code = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.CommonJS}}).outputText;
  const exports = {};
  new Function('exports', 'require', code)(exports, name => dependencies[name]);
  return exports;
}
const errors = loadSource('errors');
const { api } = loadSource('api', {'./errors': errors});
test('service errors share connection language across routes', () => {
  for (const route of ['skills', 'graphs/a/snapshots', 'skills?search=a']) {
    assert.equal(errors.requestErrorMessage(route, 503, 'Configure the Skill service API key before using Skills'), errors.connectionMessage('skill', 'unconfigured'));
    for (const status of [502, 503, 504]) assert.equal(errors.requestErrorMessage(route,status,'Service request failed'),errors.connectionMessage('skill','unavailable'));
    assert.equal(errors.requestErrorMessage(route,401,'raw upstream error'),errors.connectionMessage('skill','unauthorized'));
  }
  assert.equal(errors.requestErrorMessage('memories/search',502),errors.connectionMessage('memory','unavailable'));
});
test('local routes, validation and rate limits retain the right meaning', () => {
  assert.doesNotMatch(errors.requestErrorMessage('algorithm-visualization/report',503), /Skill/);
  assert.equal(errors.requestErrorMessage('skills',422,[{msg:'invalid'}]), 'Some request fields are invalid. Check your input and try again.');
  assert.equal(errors.requestErrorMessage('skills',409,'Skill is referenced'), 'Skill is referenced');
  assert.equal(errors.requestErrorMessage('refresh',429,'other'), errors.rateLimitMessage);
  assert.doesNotMatch(errors.requestErrorMessage('session',500,'Traceback secret'), /secret/);
});
test('transport and malformed responses become readable errors', async () => {
  const original = global.fetch;
  try {
    global.fetch = async () => {throw new TypeError('Failed to fetch');};
    await assert.rejects(api('skills'), /Cannot connect to the UI server/);
    global.fetch = async () => new Response('not json', {status:502});
    await assert.rejects(api('graphs'), /Cannot connect to the Skill service/);
    global.fetch = async () => new Response('null', {status:422});
    await assert.rejects(api('skills'), /request fields are invalid/);
    global.fetch = async () => new Response('not json');
    await assert.rejects(api('skills'), /invalid response/);
  } finally {global.fetch = original;}
});
