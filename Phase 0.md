# WJCC Newsletter — Phase 0 Log

## What I did

- Set up throwaway venv with uv, installed llama-index-readers-boarddocs and ipython
- Explored the BoardDocsReader source via IPython `??` introspection
- Discovered library is a thin wrapper around two POST endpoints
- Hit a CloudFront 403 with the library's stale headers
- Reverse-engineered working headers from browser DevTools
- Confirmed direct `requests` approach works; library not needed
- Pulled a real WJCC School Board agenda (May 5, 2026 meeting)
- Saved fixtures to ~/wjcc-fixtures/

## What I learned about BoardDocs

- Site slug: `vsba/wjcc`
- WJCC School Board committee ID: `A9HEJH3AA9D7`
- Two endpoints I need:
  - `POST /BD-GetMeetingsList?open` → JSON list of meetings
  - `POST /PRINT-AgendaDetailed` → HTML for a specific meeting
- Modern browser-style headers required; stale UAs get blocked by CloudFront WAF
- One empty `{}` record in the meetings list — filter on `m.get("unique")`
- `current=1` flag exists but doesn't mean "upcoming" — use date comparison instead
- Completed agendas include Motion & Voting blocks with vote tallies
- ~74k characters for a full agenda; well within Sonnet's context window

## Decisions made

- Drop llama-index dependency entirely; use requests + bs4 + html2text directly
- Update Phase 1 Claude Code prompt before starting build


## Open questions

- How far in advance are agendas posted? **Set a reminder to check**
When are agendas updated with the vote outcomes?
Would it be better to focus on following up with vote outcomes asap as the main newsletter output than doing a preview? Would it be easy enough to do both?
Is there a way to give the LLM more context, like searching for any news stories about items on the agenda to weigh their importance? Probably a future add but curious
When I covered the school board a couple years ago, in my experience they would have disagreements mainly during the working meetings, come to a consensus, and then all vote the same way during the regular meeting, although that definitely wasn't always the case like on controversial items.
Will definitely need to experiment and get feedback from subscribers and others like the WJCC Parents & Community FB group
How will the workflow work going from my code extracting the info to a finished Substack post?
Down the road will I offer a paid tier of the newsletter? What would that look like?

## Fixtures saved

- ~/wjcc-fixtures/sample-agenda.html (raw)
- ~/wjcc-fixtures/sample-agenda.md (html2text output)
- ~/wjcc-fixtures/sample-meeting-meta.json
- ~/wjcc-fixtures/working-headers.json
