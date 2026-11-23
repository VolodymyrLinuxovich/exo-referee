// exo-prior-work: an Apify Actor that finds prior work for an agent's hypothesis.
//
// Input:  { "queries": ["centroid offset false positive kepler", ...], "maxPerQuery": 5, "categories": ["astro-ph.EP", ...] }
// Output: one dataset item per paper: { query, source, title, authors, published, url, abstract }
//
// arXiv first: its public Atom API through Crawlee's CheerioCrawler (XML mode), one request at a time
// with a pause, through Apify Proxy with a session pool so a rate-limited IP is retired and retried.
// If arXiv still refuses or a search finds nothing, the same query goes to OpenAlex, an open index.
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

const words = (q, keep) => q.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").split(/\s+/).filter((w) => w.length > 2).slice(0, keep);

function arxivUrl(q, keep = 5) {
    const terms = words(q, keep).map((w) => `all:${w}`).join(" AND ");
    const cats = categories.map((c) => `cat:${c}`).join(" OR ");
    const params = new URLSearchParams({ search_query: cats ? `(${terms}) AND (${cats})` : terms, start: "0",
        max_results: String(maxPerQuery), sortBy: "relevance" });
    return `https://export.arxiv.org/api/query?${params}`;
}

function openAlexUrl(q) {
    const params = new URLSearchParams({ search: words(q, 6).join(" "), "per-page": String(maxPerQuery),
        filter: "type:article", select: "title,authorships,publication_date,doi,id,locations,abstract_inverted_index" });
    return `https://api.openalex.org/works?${params}`;
}

// OpenAlex stores abstracts as word -> positions; rebuild the text.
function abstractFrom(inv) {
    if (!inv) return "";
    const pos = [];
    for (const [w, ps] of Object.entries(inv)) for (const p of ps) pos[p] = w;
    return pos.join(" ");
}

// Apify Proxy on the platform; none when running locally without one.
let proxyConfiguration;
try {
    proxyConfiguration = await Actor.createProxyConfiguration();
} catch (err) {
    log.warning(`No Apify Proxy, using direct connections: ${err.message}`);
}

const fallback = async (crawler, request) => {
    const q = request.userData.query;
    await crawler.addRequests([{ url: openAlexUrl(q), userData: { query: q, source: "openalex" }, uniqueKey: `${q}#openalex` }]);
};

const crawler = new CheerioCrawler({
    proxyConfiguration,
    useSessionPool: true,
    maxConcurrency: 1,                 // arXiv: one request at a time
    sameDomainDelaySecs: 3,            // and a pause between them
    maxRequestRetries: 4,
    navigationTimeoutSecs: 20,
    additionalMimeTypes: ["application/atom+xml", "application/xml", "text/xml", "application/json"],
    preNavigationHooks: [(_ctx, gotOptions) => {
        gotOptions.headers = { ...gotOptions.headers, "user-agent": "exo-prior-work (research referee)" };
    }],
    async requestHandler({ request, body, addRequests, crawler: self }) {
        const { query, source = "arxiv", relaxed } = request.userData;
        let papers;
        if (source === "openalex") {
            const data = JSON.parse(body.toString());
            papers = (data.results ?? []).map((w) => {
                const arxiv = (w.locations ?? []).map((l) => l.landing_page_url).find((u) => u && u.includes("arxiv.org"));
                return { query, source, title: (w.title ?? "").trim(), authors: (w.authorships ?? []).map((a) => a.author?.display_name).filter(Boolean),
                         published: w.publication_date ?? "", url: arxiv ?? w.doi ?? w.id, abstract: abstractFrom(w.abstract_inverted_index).slice(0, 600) };
            }).filter((p) => p.title);
        } else {
            const { load } = await import("cheerio");
            const $ = load(body.toString(), { xmlMode: true });
            papers = $("entry").map((_, e) => {
                const el = $(e);
                return { query, source, title: el.find("title").first().text().replace(/\s+/g, " ").trim(),
                         authors: el.find("author > name").map((__, a) => $(a).text().trim()).get(),
                         published: el.find("published").first().text().slice(0, 10), url: el.find("id").first().text().trim(),
                         abstract: el.find("summary").first().text().replace(/\s+/g, " ").trim().slice(0, 600) };
            }).get().filter((p) => p.title);
        }
        log.info(`${papers.length} papers for "${query}" from ${source}${relaxed ? " (relaxed)" : ""}`);
        await Actor.pushData(papers);
        if (!papers.length && source === "arxiv" && !relaxed) {
            // too strict: retry once with the three strongest words
            await addRequests([{ url: arxivUrl(query, 3), userData: { query, relaxed: true }, uniqueKey: `${query}#relaxed`, maxRetries: 1 }]);
        } else if (!papers.length && source === "arxiv") {
            await fallback(self, request);
        }
    },
    async failedRequestHandler({ request, crawler: self }, error) {
        log.warning(`${request.userData.source ?? "arxiv"} failed for "${request.userData.query}": ${error.message}`);
        if ((request.userData.source ?? "arxiv") === "arxiv") await fallback(self, request);
    },
});

// arXiv gets one retry (a fresh proxy session); after that OpenAlex answers, so a search never stalls
await crawler.run(queries.map((query) => ({ url: arxivUrl(query), userData: { query }, uniqueKey: query, maxRetries: 1 })));
await Actor.exit();
