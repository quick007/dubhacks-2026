/** Published findings on speed, findability and sales, each checked against its source. */
export const findings = [
  {
    value: "0.6%",
    claim: "Bing revenue loss per 100 ms added",
    method: "Randomized experiment at Bing: 10% of users slowed 100 ms, 10% slowed 250 ms, 2 weeks",
    reference:
      "R. Kohavi, A. Deng, B. Frasca, T. Walker, Y. Xu, N. Pohlmann. Online Controlled Experiments at Large Scale. KDD 2013, §3.4.",
    url: "https://exp-platform.com/large-scale/",
  },
  {
    value: "+8.4%",
    claim: "retail conversions in the mobile-speed study",
    method: "37 retail, travel, luxury and lead generation brands, 4 weeks. Order value rose 9.2%",
    reference: "Deloitte Digital. Milliseconds Make Millions. 2020. Commissioned by Google.",
    url: "https://web.dev/case-studies/milliseconds-make-millions",
  },
  {
    value: "+32%",
    claim: "higher bounce chance at 3 s versus 1 s",
    method: "Mobile landing pages across industries and countries",
    reference: "Google/SOASTA Research. Think with Google, 2017.",
    url: "https://www.thinkwithgoogle.com/data/page-load-time-statistics/",
  },
  {
    value: "48%",
    claim: "of shoppers buy elsewhere after a failed search",
    method: "Survey of 10,000+ shoppers. 76% say a failed search cost the site the sale",
    reference:
      "Google Cloud and The Harris Poll. Search Abandonment in Retail. 2021. Vendor survey.",
    url: "https://cloud.google.com/blog/topics/retail/search-abandonment-impacts-retail-sales-brand-loyalty",
  },
];
