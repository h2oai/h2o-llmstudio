"use strict";(self.webpackChunksite=self.webpackChunksite||[]).push([[889],{5680:(e,t,n)=>{n.d(t,{xA:()=>u,yg:()=>g});var a=n(6540);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function r(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?r(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):r(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,o=function(e,t){if(null==e)return{};var n,a,o={},r=Object.keys(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var p=a.createContext({}),s=function(e){var t=a.useContext(p),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},u=function(e){var t=s(e.components);return a.createElement(p.Provider,{value:t},e.children)},c="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},m=a.forwardRef((function(e,t){var n=e.components,o=e.mdxType,r=e.originalType,p=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),c=s(n),m=o,g=c["".concat(p,".").concat(m)]||c[m]||d[m]||r;return n?a.createElement(g,i(i({ref:t},u),{},{components:n})):a.createElement(g,i({ref:t},u))}));function g(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var r=n.length,i=new Array(r);i[0]=m;var l={};for(var p in t)hasOwnProperty.call(t,p)&&(l[p]=t[p]);l.originalType=e,l[c]="string"==typeof e?e:o,i[1]=l;for(var s=2;s<r;s++)i[s]=n[s];return a.createElement.apply(null,i)}return a.createElement.apply(null,n)}m.displayName="MDXCreateElement"},3824:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>p,contentTitle:()=>i,default:()=>d,frontMatter:()=>r,metadata:()=>l,toc:()=>s});var a=n(8168),o=(n(6540),n(5680));const r={},i="Evaluate model using an AI judge",l={unversionedId:"guide/experiments/evaluate-model-using-llm",id:"guide/experiments/evaluate-model-using-llm",title:"Evaluate model using an AI judge",description:"H2O LLM Studio provides the option to use an AI Judge like ChatGPT or a local LLM deployment to evaluate a fine-tuned model.",source:"@site/docs/guide/experiments/evaluate-model-using-llm.md",sourceDirName:"guide/experiments",slug:"/guide/experiments/evaluate-model-using-llm",permalink:"/h2o-llmstudio/guide/experiments/evaluate-model-using-llm",draft:!1,tags:[],version:"current",frontMatter:{},sidebar:"defaultSidebar",previous:{title:"Import a model to h2oGPT",permalink:"/h2o-llmstudio/guide/experiments/import-to-h2ogpt"},next:{title:"FAQs",permalink:"/h2o-llmstudio/faqs"}},p={},s=[],u={toc:s},c="wrapper";function d(e){let{components:t,...r}=e;return(0,o.yg)(c,(0,a.A)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,o.yg)("h1",{id:"evaluate-model-using-an-ai-judge"},"Evaluate model using an AI judge"),(0,o.yg)("p",null,"H2O LLM Studio provides the option to use an AI Judge like ChatGPT or a local LLM deployment to evaluate a fine-tuned model. "),(0,o.yg)("p",null,"Follow the instructions below to specify a local LLM to evaluate the responses of the fine-tuned model."),(0,o.yg)("ol",null,(0,o.yg)("li",{parentName:"ol"},(0,o.yg)("p",{parentName:"li"},"Have an endpoint running of the local LLM deployment, which supports the OpenAI API format; specifically the ",(0,o.yg)("a",{parentName:"p",href:"https://platform.openai.com/docs/guides/text-generation/chat-completions-api"},"Chat Completions API"),".")),(0,o.yg)("li",{parentName:"ol"},(0,o.yg)("p",{parentName:"li"},"Start the H2O LLM Studio server with the following environment variable that points to the endpoint. "),(0,o.yg)("pre",{parentName:"li"},(0,o.yg)("code",{parentName:"pre"},'OPENAI_API_BASE="http://111.111.111.111:8000/v1"\n'))),(0,o.yg)("li",{parentName:"ol"},(0,o.yg)("p",{parentName:"li"},"Once H2O LLM Studio is up and running, click ",(0,o.yg)("strong",{parentName:"p"},"Settings")," on the left navigation panel to validate that the endpoint is being used correctly. The ",(0,o.yg)("strong",{parentName:"p"},"Use OpenAI API on Azure")," setting must be set to Off, and the environment variable that was set above should be the ",(0,o.yg)("strong",{parentName:"p"},"OpenAI API Endpoint")," value as shown below.\n",(0,o.yg)("img",{alt:"set-endpoint",src:n(9958).A,width:"2852",height:"1820"})),(0,o.yg)("admonition",{parentName:"li",type:"info"},(0,o.yg)("p",{parentName:"admonition"},"Note that changing the value of this field here on the GUI has no effect. This is only for testing the correct setting of the environment variable."))),(0,o.yg)("li",{parentName:"ol"},(0,o.yg)("p",{parentName:"li"},"Run an experiment using ",(0,o.yg)("inlineCode",{parentName:"p"},"GPT")," as the ",(0,o.yg)("strong",{parentName:"p"},"Metric")," and the relevant model name available at your endpoint as the ",(0,o.yg)("strong",{parentName:"p"},"Metric Gpt Model"),".\n",(0,o.yg)("img",{alt:"set-metric-model",src:n(1370).A,width:"3144",height:"1800"}))),(0,o.yg)("li",{parentName:"ol"},(0,o.yg)("p",{parentName:"li"},"Validate that it is working as intended by checking the logs. Calls to the LLM judge should now be directed to your own LLM endpoint.\n",(0,o.yg)("img",{alt:"local-llm-judge-logs",src:n(9096).A,width:"2524",height:"100"})))))}d.isMDXComponent=!0},9096:(e,t,n)=>{n.d(t,{A:()=>a});const a=n.p+"assets/images/local-llm-judge-logs-674c978e8e9483339607432f886b2de6.png"},9958:(e,t,n)=>{n.d(t,{A:()=>a});const a=n.p+"assets/images/set-endpoint-7931ae1b3c494a334b064a67c0a9d9d0.png"},1370:(e,t,n)=>{n.d(t,{A:()=>a});const a=n.p+"assets/images/set-metric-model-db5fc57f4cb22f677305acfa6fc0529c.png"}}]);