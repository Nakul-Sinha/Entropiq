// @vitest-environment node
import { describe, expect, it } from "vitest";

import { canonicalRepoUrl, repoUrlError } from "./repo-url";

describe("repoUrlError", () => {
  it.each([
    "https://github.com/example/inventory-service",
    "https://github.com/example/inventory-service.git",
    "https://github.com/example/inventory-service/",
    "https://gitlab.com/group/project",
    "https://GitHub.com/Example/Repo.git/",
    "https://github.com:443/example/repo",
    "  https://github.com/a_b/c.d-e  ",
  ])("accepts %s", (url) => {
    expect(repoUrlError(url)).toBeNull();
  });

  it.each([
    ["", "Enter a repository URL."],
    ["   ", "Enter a repository URL."],
    ["http://github.com/example/repo", "Use an https URL."],
    ["ssh://git@github.com/example/repo", "Use an https URL."],
    ["git@github.com:example/repo.git", "Use an https URL."],
    [
      "https://user:secret@github.com/example/repo",
      "Remove the user name or password from the URL.",
    ],
    ["https://github.com:8443/example/repo", "Use the default https port."],
    ["https://bitbucket.org/example/repo", "Only GitHub and GitLab repositories are supported."],
    [
      "https://github.com/example/repo/tree/main",
      "Use the repository URL, like https://github.com/owner/name.",
    ],
    ["https://github.com/example", "Use the repository URL, like https://github.com/owner/name."],
    ["https://github.com/../repo", null],
    [
      "https://github.com/example/repo?tab=readme",
      "Remove the query string or fragment from the URL.",
    ],
    ["https://github.com/example/repo#readme", "Remove the query string or fragment from the URL."],
    ["https://github.com/exa mple/repo", null],
  ])("rejects %s", (url, message) => {
    const error = repoUrlError(url);
    expect(error).not.toBeNull();
    if (message !== null) expect(error).toBe(message);
  });
});

describe("canonicalRepoUrl", () => {
  it("drops .git and the trailing slash", () => {
    expect(canonicalRepoUrl("https://github.com/example/inventory-service.git/")).toBe(
      "https://github.com/example/inventory-service",
    );
    expect(canonicalRepoUrl(" https://gitlab.com/Group/Project ")).toBe(
      "https://gitlab.com/Group/Project",
    );
  });

  it("lower-cases GitHub URLs, which ignore case", () => {
    expect(canonicalRepoUrl("https://GitHub.com/Example/Repo")).toBe(
      "https://github.com/example/repo",
    );
  });

  it("throws for an invalid URL", () => {
    expect(() => canonicalRepoUrl("http://x")).toThrow("Use an https URL.");
  });
});
