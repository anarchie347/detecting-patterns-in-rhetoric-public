// designed to run on ec2
// need to install @aws-sdk/client-s3

import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import {writeFile, mkdir, access} from "fs/promises";
import { argv } from "process";
import sanitize from "sanitize-filename";

const s3Client = new S3Client();

const BUCKET_NAME = "";
async function handler() {
  const { parsedArticle, rawArticle, articleName } =
    await wikipedia_random_article();
  await writeArticle(articleName, parsedArticle, rawArticle);
}

async function wikipedia_random_article() {
// console.log(await fetch_wikipedia({action: "query", format: "json", list: "random", rnnamespace: 0, rnlimit: 1}));
// return;
  const articleName = (
    await fetch_wikipedia({
      action: "query",
      format: "json",
      list: "random",
      rnnamespace: 0,
      rnlimit: 1,
    })
  ).query.random[0].title;
  const articlePages = (
    await fetch_wikipedia({
      action: "query",
      prop: "extracts", //no metadata
      explaintext: 1, //plaintext, no html
      titles: articleName,
      format: "json",
    })
  ).query.pages;
  const rawArticle =
    Object.keys(articleName).length > 0
      ? articlePages[Object.keys(articlePages)[0]].extract
      : "";
  const parsedArticle = parse_sections(rawArticle, articleName);
  return { parsedArticle, rawArticle, articleName };
}

async function fetch_wikipedia(params) {
  const endpoint = "https://en.wikipedia.org/w/api.php";

  const paramStr = new URLSearchParams(params).toString();
  const resp = await fetch(endpoint + "?" + paramStr);
  const text = await resp.text();
  return JSON.parse(text);
}

/**
 *
 * @param {string} rawtext
 */
function parse_sections(rawtext, title) {
  const lines = rawtext.split(/\r?\n/).map((x) => x.trim());
  const sections = [];

  let lastTitle = -1;
  let lastHeader = title;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith("==") && lines[i].endsWith("==")) {
      // some type of heading
      const lastSection = lines.slice(lastTitle + 1, i).join("\n");
      sections.push({ header: lastHeader, text: lastSection });
      lastHeader = lines[i].replace(/=/g, " ").trim();
      lastTitle = i;
    }
  }
  const lastSection = lines.slice(lastTitle + 1).join("");
  sections.push({ header: lastHeader, text: lastSection });
  return sections;
}

async function writeArticle(title, parsed, raw) {
    //writeToS3(title, parsed, raw);
    writeToFile(title, parsed, raw);
}

async function writeToFile(title, parsed, raw) {
    const sanTitle = sanitize(title);
    if (await fsExists(`data/wiki-${sanTitle}`))
        return
    await mkdir(`data/wiki-${sanTitle}`);
    await writeFile(`data/wiki-${sanTitle}/full.txt`, raw);
    await Promise.all(
        parsed.map(({header, text}) =>
            writeFile(`data/wiki-${sanTitle}/section-${sanitize(header)}.txt`, text)
        )
    );
}

async function writeToS3(title, parsed, raw) {
  const sanTitle = sanitize(title);
  const pushRawCmd = new PutObjectCommand({
    Bucket: BUCKET_NAME,
    Key: `wiki-${sanTitle}/full.txt`,
    Body: raw,
  });
  await s3Client.send(cmd);

  await Promise.all(
    parsed.map(async ({ header, text }) => {
      const pushCmd = new PutObjectCommand({
        Bucket: BUCKET_NAME,
        Key: `wiki-${sanTitle}/section-${sanitize(header)}.txt`,
        Body: text,
      });
      await s3Client.send(pushCmd);
    }),
  );
}

async function fsExists(path) {
    try {
        await access(path);
        return true;
    } catch {
        return false;
    }
}

// MAIN CODE

const repeats = argv[2]
const delayS = argv[3]

if (!await fsExists("data"))
    await mkdir("data");

for (let i = 0; i < repeats; i++) {
    try {
        await handler();
        console.log("COMPLETED #" + i);
    } catch (err) {
        console.log("FAILED #" + i);
    }
    await new Promise((res,rej) => setTimeout(res, delayS * 1000));
}

console.log("DONE");
