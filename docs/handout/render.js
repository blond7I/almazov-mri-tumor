const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + __dirname + '/handout.html', { waitUntil: 'load' });
  await p.waitForTimeout(400);
  const h = await p.evaluate(() => document.body.scrollHeight);
  console.log('body scrollHeight px', h, '(A4 = 1123 at 96dpi)');
  await p.pdf({ path: 'doctor_handout.pdf', format: 'A4', printBackground: true });
  await p.setViewportSize({ width: 794, height: 1123 });
  await p.screenshot({ path: 'preview.png', fullPage: true, scale: 'css' });
  await b.close();
})();
