(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.KickbaseUtils = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  function minutes(entry) {
    return parseInt(entry && entry.minutes, 10) || 0;
  }

  /** Letzter tatsaechlich gespielter Tag, ohne vorab angelegte Null-Zeilen. */
  function latestPlayedDay(players) {
    let latest = 0;
    (players || []).forEach(player => {
      (player.performance || []).forEach(entry => {
        if (minutes(entry) > 0 || Number(entry.points) !== 0) {
          latest = Math.max(latest, Number(entry.day) || 0);
        }
      });
    });
    return latest;
  }

  return { latestPlayedDay };
}));
