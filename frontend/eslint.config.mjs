import { fixupConfigRules, fixupPluginRules } from "@eslint/compat";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import jsxA11Y from "eslint-plugin-jsx-a11y";
import prettier from "eslint-plugin-prettier";
import unusedImports from "eslint-plugin-unused-imports";
import _import from "eslint-plugin-import";
import simpleImportSort from "eslint-plugin-simple-import-sort";
import path from "node:path";
import { fileURLToPath } from "node:url";
import js from "@eslint/js";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const compat = new FlatCompat({
    baseDirectory: __dirname,
    recommendedConfig: js.configs.recommended,
    allConfig: js.configs.all
});

export default [...fixupConfigRules(compat.extends(
    "next/core-web-vitals",
    "plugin:react/recommended",
    "plugin:react-hooks/recommended",
    "plugin:jsx-a11y/recommended",
    "plugin:prettier/recommended",
)), {
    plugins: {
        react: fixupPluginRules(react),
        "react-hooks": fixupPluginRules(reactHooks),
        "jsx-a11y": fixupPluginRules(jsxA11Y),
        prettier: fixupPluginRules(prettier),
        "unused-imports": unusedImports,
        import: fixupPluginRules(_import),
        "simple-import-sort": simpleImportSort,
    },

    settings: {
        react: {
            version: "detect",
        },
    },

    rules: {
        "react/react-in-jsx-scope": "off",
        "react/prop-types": "off",

        "prettier/prettier": ["error", {
            singleQuote: true,
            trailingComma: "es5",
            endOfLine: "auto",
        }],

        "unused-imports/no-unused-imports": "error",
        "import/order": ["error"],
        "simple-import-sort/imports": "error",
        "simple-import-sort/exports": "error",
    },
}];