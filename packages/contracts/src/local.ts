/** Local development routing only; HTTP wire schemas remain unchanged. */
export const LOCAL_PROXY_PORT = 1356;
export const LOCAL_ORIGINS = {
  dashboard: `http://dubhacks.localhost:${LOCAL_PROXY_PORT}`,
  api: `http://api.dubhacks.localhost:${LOCAL_PROXY_PORT}`,
  storefront: `http://shop.dubhacks.localhost:${LOCAL_PROXY_PORT}`,
} as const;
