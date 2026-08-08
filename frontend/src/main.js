const app = document.querySelector('#app');

const steps = [
  {
    title: 'Connect to backend',
    detail: 'Decide how the UI will fetch analysis runs, reports, and datasets.',
  },
  {
    title: 'Build the workspace',
    detail: 'Add navigation, insight panels, and a dataset explorer.',
  },
  {
    title: 'Wire interactions',
    detail: 'Let users submit questions, review findings, and export results.',
  },
];

app.innerHTML = `
  <div class="stack">
    ${steps
      .map(
        (step, index) => `
          <article class="step-card">
            <div class="step-index">0${index + 1}</div>
            <div>
              <h3>${step.title}</h3>
              <p>${step.detail}</p>
            </div>
          </article>
        `,
      )
      .join('')}
  </div>
`;
