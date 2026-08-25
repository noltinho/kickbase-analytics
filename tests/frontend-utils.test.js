'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { latestPlayedDay, defaultMinAppearances } = require('../frontend-utils.js');

test('latestPlayedDay ignores future zero rows', () => {
  const players = [{ performance: [
    { day: 5, points: 80, minutes: "90'" },
    { day: 6, points: 0, minutes: "0'" },
    { day: 34, points: 0, minutes: "0'" },
  ] }];
  assert.equal(latestPlayedDay(players), 5);
});

test('latestPlayedDay accepts played rows with non-zero points', () => {
  const players = [{ performance: [{ day: 3, points: -5, minutes: "0'" }] }];
  assert.equal(latestPlayedDay(players), 3);
});

test('defaultMinAppearances uses half the played matchdays, rounded up', () => {
  const players = [{ performance: [
    { day: 5, points: 80, minutes: "90'" },
    { day: 34, points: 0, minutes: "0'" },
  ] }];
  assert.equal(defaultMinAppearances(players), 3);
  assert.equal(defaultMinAppearances([]), 0);
});

test('individual season filter labels both entries as seasons', () => {
  const context = vm.createContext({
    URLSearchParams,
    location: { pathname: '/teampunkte.html', search: '' },
    localStorage: { getItem: () => null, setItem: () => {} },
    fetch: () => Promise.reject(new Error('not used')),
  });
  vm.runInContext(fs.readFileSync('common.js', 'utf8'), context);
  const result = vm.runInContext(`
    SEASONS = {
      current: '2627',
      seasons: [
        { key: '2627', label: '26/27', suffix: '' },
        { key: '2526', label: '25/26', suffix: '_2526' }
      ]
    };
    (() => {
      const el = { innerHTML: '', value: '' };
      const value = fillSeasonSelect(el, 'agg', false, true);
      return { html: el.innerHTML, value };
    })();
  `, context);
  assert.equal(result.value, '2627');
  assert.match(result.html, />Saison 26\/27</);
  assert.match(result.html, />Saison 25\/26</);
  assert.doesNotMatch(result.html, /Vorsaison|Archiv/);
});
