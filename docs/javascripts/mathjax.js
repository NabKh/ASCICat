// MathJax configuration for ASCICat documentation
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams'
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  svg: {
    fontCache: 'global'
  },
  loader: {
    load: ['[tex]/ams', '[tex]/color', '[tex]/boldsymbol']
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
