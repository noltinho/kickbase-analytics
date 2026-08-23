'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { latestPlayedDay } = require('../frontend-utils.js');

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
