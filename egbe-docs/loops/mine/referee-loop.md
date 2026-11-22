# Loop: Exo Referee, keep improving and keep checking

**Decision this loop makes, every cycle:** which agent-proposed change to the planet classifier becomes
the new model, and whether the referee itself can still be trusted to decide that.

**Cadence:** every 6 hours on this Mac (`loop/com.exoreferee.loop.plist`). The sandbox uses macOS
`sandbox-exec`, so the loop runs where the sandbox works.

**Each cycle** (`loop/tick.py`):
1. Check the checker: `exo/selftest.py` plants three cheats, runs an honest control and attacks the
   sandbox. Any failure halts the loop before an agent claim is judged.
2. Process new data: pull today's NASA dispositions and score our ranking of the 1,977 unresolved
   Kepler candidates on any that NASA has resolved since the 2026-11-22 snapshot.
3. Background work: one guarded round. Agents research prior work through the `exo-prior-work`
   Apify Actor, propose one change each, and the Referee judges it. Only PROMOTED code moves the model.
4. Deploy updates: rebuild the site data from the new verdicts and publish (`loop/deploy.sh`).
5. Report: `egbe-docs/reports/loop-<time>.md`, synced to Egbe.

**Human on the loop:** reads the reports. Steps in when a cycle halts on a failed self-test, when a
promotion looks surprising, or when spend needs changing (`LOOP_PROPOSERS`, default 2 agents per cycle).

**Success signal:** promotions that hold on hidden stars, and our ranking's AUC on answers NASA
publishes after we made it.
