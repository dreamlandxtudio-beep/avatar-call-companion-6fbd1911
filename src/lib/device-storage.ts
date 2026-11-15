export async function requestDeviceStorage() {
  if (!navigator.storage) return false;
  return navigator.storage.persist ? navigator.storage.persist() : false;
}
export async function deviceStorageInfo() {
  const estimate = await navigator.storage?.estimate();
  const persistent = await navigator.storage?.persisted?.();
  return { usedMB: Math.round((estimate?.usage ?? 0) / 1024 / 1024), persistent: !!persistent };
}
