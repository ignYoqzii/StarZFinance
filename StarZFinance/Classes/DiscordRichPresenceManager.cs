using DiscordRPC;

namespace StarZFinance.Classes
{
    public static class DiscordRichPresenceManager
    {
        private static readonly string ClientId = "1329895465977774182";
        private static DiscordRpcClient? discordClient;

        public static DiscordRpcClient DiscordClient
        {
            get
            {
                discordClient ??= new DiscordRpcClient(ClientId);
                return discordClient;
            }
        }

        public static void SetPresence(string details = "")
        {
            try
            {
                string state = ConfigManager.GetDiscordRPCIdleStatus();
                DiscordClient.SetPresence(new RichPresence
                {
                    State = state,
                    Details = details,
                    Timestamps = Timestamps.Now,
                    Assets = new Assets
                    {
                        LargeImageKey = "starz",
                        LargeImageText = "StarZ Finance",
                        SmallImageKey = "finance"
                    }
                });
            }
            catch (Exception ex)
            {
                LogsManager.Log($"{ex.Message}", "DiscordClient.txt");
            }
        }

        public static void IdlePresence(string state)
        {
            try
            {
                DiscordClient.UpdateState(state);
                DiscordClient.UpdateDetails("");
            }
            catch (Exception ex)
            {
                LogsManager.Log($"{ex.Message}", "DiscordClient.txt");
            }
        }

        public static void TerminatePresence()
        {
            if (DiscordClient.IsDisposed) return;
            try
            {
                DiscordClient.ClearPresence();
                DiscordClient.Deinitialize();
                DiscordClient.Dispose();
            }
            catch (Exception ex)
            {
                LogsManager.Log($"{ex.Message}", "DiscordClient.txt");
            }
        }
    }
}