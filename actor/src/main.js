// exo-prior-work: an Apify Actor that finds prior work on arXiv for an agent's hypothesis.
//
// Input:  { "queries": ["centroid offset false positive kepler", ...], "maxPerQuery": 5, "categories": ["astro-ph.EP", "astro-ph.IM"] }
// Output: one dataset item per paper: { query, title, authors, published, url, abstract }
//
// It reads arXiv's public Atom API with Crawlee's CheerioCrawler (XML mode), one request at a time
// with a pause between requests, as arXiv asks of API clients.
import { Actor } from "apify";
import { CheerioCrawler, log } from "crawlee";

await Actor.init();

const input = (await Actor.getInput()) ?? {};
const queries = (input.queries ?? []).filter(Boolean);
const maxPerQuery = Math.min(Math.max(input.maxPerQuery ?? 5, 1), 25);
const categories = input.categories ?? ["astro-ph.EP", "astro-ph.IM", "astro-ph.SR"];

if (!queries.length) {
    log.warning("No queries given; nothing to search.");
    await Actor.exit();
}

// arXiv search syntax: words AND-ed in all fields, limited to the given categories.
function searchUrl(q, keep = 5) {
    const words = q.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").split(/\s+/).filter((w) => w.length > 2).slice(0, keep);
    const terms = words.map((w) => `all:${w}`).join(" AND ");
    const cats = categories.map((c) => `cat:${c}`).join(" OR ");
    const search = cats ? `(${terms}) AND (${cats})` : terms;
    const params = new URLSearchParams({ search_query: search, start: "0", max_results: String(maxPerQuery), sortBy: "relevance" });
    return `https://export.arxiv.org/api/query?${params}`;
}

const crawler = new CheerioCrawler({
    maxConcurrency: 1,                 // arXiv: one request at a time
    sameDomainDelaySecs: 3,            // and a pause between them
    maxRequestRetries: 2,
    additionalMimeTypes: ["application/atom+xml", "application/xml", "text/xml"],
    preNavigationHooks: [(_ctx, gotOptions) => {
        gotOptions.headers = { ...gotOptions.headers, "user-agent": "exo-prior-work (research referee)" };
    }],
    async requestHandler({ request, body, addRequests }) {
        const { load } = await import("cheerio");
        const $ = load(body.toString(), { xmlMode: true });
        const papers = $("entry").map((_, e) => {
            const el = $(e);
            return {
                query: request.userData.query,
                title: el.find("title").first().text().replace(/\s+/g, " ").trim(),
                authors: el.find("author > name").map((__, a) => $(a).text().trim()).get(),
                published: el.find("published").first().text().slice(0, 10),
                url: el.find("id").first().text().trim(),
                abstract: el.find("summary").first().text().replace(/\s+/g, " ").trim().slice(0, 600),
            };
        }).get().filter((p) => p.title);
        log.info(`${papers.length} papers for "${request.userData.query}"${request.userData.relaxed ? " (relaxed)" : ""}`);
        if (!papers.length && !request.userData.relaxed) {
            // too strict: retry once with the three strongest words
            const q = request.userData.query;
            await addRequests([{ url: searchUrl(q, 3), userData: { query: q, relaxed: true }, uniqueKey: `${q}#relaxed` }]);
        }
        await Actor.pushData(papers);
    },
    failedRequestHandler({ request }, error) {
        log.error(`Search failed for "${request.userData.query}": ${error.message}`);
    },
});

await crawler.run(queries.map((query) => ({ url: searchUrl(query), userData: { query }, uniqueKey: query })));
await Actor.exit();
