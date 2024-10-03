export default {
  base: "./",
  build: {
    lib: {
      entry: "./src/main.js",
      name: "breasties_2",
      formats: ["umd"],
      fileName: "breasties_2",
    },
    rollupOptions: {
      external: ["vue"],
      output: {
        globals: {
          vue: "Vue",
        },
      },
    },
    outDir: "../breasties_2/module/serve",
    assetsDir: ".",
  },
};
