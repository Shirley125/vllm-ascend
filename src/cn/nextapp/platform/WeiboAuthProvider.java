package cn.nextapp.platform;

import java.util.List;
import java.util.Map;

import javax.servlet.http.HttpServletRequest;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.brickred.socialauth.AbstractProvider;
import org.brickred.socialauth.AuthProvider;
import org.brickred.socialauth.Contact;
import org.brickred.socialauth.Permission;
import org.brickred.socialauth.Profile;
import org.brickred.socialauth.exception.SocialAuthException;
import org.brickred.socialauth.util.AccessGrant;
import org.brickred.socialauth.util.OAuthConfig;
import org.brickred.socialauth.util.Response;
import org.brickred.socialauth.util.SocialAuthUtil;

import weibo4j.User;
import weibo4j.Weibo;
import weibo4j.WeiboException;
import weibo4j.http.AccessToken;
import weibo4j.http.RequestToken;


/**
 * 实现新浪帐号登录
 * @author Winter Lau
 * @date 2011-12-28 下午5:42:15
 */
public class WeiboAuthProvider extends AbstractProvider implements AuthProvider {

	private static final long serialVersionUID = -6768964716431375260L;
	
	private final static String ID = "weibo";
    private final Log LOG = LogFactory.getLog(WeiboAuthProvider.class);

    private OAuthConfig config;
    private Profile userProfile;
    private RequestToken resToken;

	public WeiboAuthProvider(final OAuthConfig providerConfig) throws Exception {
		config = providerConfig;
		Weibo.CONSUMER_KEY = config.get_consumerKey();
		Weibo.CONSUMER_SECRET = config.get_consumerSecret();
	}

	@Override
	public Response api(final String url, final String methodType,
            final Map<String, String> params,
            final Map<String, String> headerParams, final String body) throws Exception {
		Response response = null;
		return response;

	}

	@Override
	public AccessGrant getAccessGrant() {
		return null;
	}

	@Override
	public List<Contact> getContactList() throws Exception {
		return null;
	}

	/**
	 * 用户授权验证
	 * @param backUrl
	 * @return
	 */
	public RequestToken request(String backUrl) throws Exception {
		RequestToken requestToken = null;
		try {
			System.setProperty("weibo4j.oauth.consumerKey", Weibo.CONSUMER_KEY);
			System.setProperty("weibo4j.oauth.consumerSecret",Weibo.CONSUMER_SECRET);
			
			Weibo weibo = new Weibo();
			requestToken = weibo.getOAuthRequestToken(backUrl);
		} catch (Exception e) {
			LOG.error("Failed to get RequestToken.", e);
			throw new SocialAuthException("Failed to get RequestToken.",e);
		}
		return requestToken;
	}
	
	@Override
	public String getLoginRedirectURL(final String successUrl) throws Exception {
		resToken = request(successUrl);
		return resToken.getAuthorizationURL();
	}

	@Override
	public String getProviderId() {
		return ID;
	}

	@Override
	public Profile getUserProfile() throws Exception {
		return userProfile;
	}

	@Override
	public void logout() {
		resToken = null;
	}

	@Override
	public void setAccessGrant(AccessGrant accessGrant) throws Exception {
		
	}

	@Override
	public void setPermission(Permission p) {

	}

	@Override
	public void updateStatus(String arg0) throws Exception {

	}

	@Override
	public Profile verifyResponse(HttpServletRequest request) throws Exception {
		Map<String, String> params = SocialAuthUtil
				.getRequestParametersMap(request);
		return doVerifyResponse(params);
	}

	@Override
	public Profile verifyResponse(Map<String, String> requestParams) throws Exception {
		return doVerifyResponse(requestParams);
	}

	private Profile doVerifyResponse(final Map<String, String> requestParams) throws Exception {
		String verifier=requestParams.get("oauth_verifier");

		if(verifier!=null)
		{
			AccessToken accessToken = requstAccessToken(resToken,verifier);
			if(accessToken!=null)
			{			
				Profile p = new Profile();
				p.setEmail(accessToken.getToken());
				p.setFirstName(accessToken.getScreenName());
				p.setValidatedId(accessToken.getTokenSecret());
				p.setProviderId(getProviderId());
				try {
					Weibo weibo = new Weibo();
					weibo.setToken(accessToken);
					User user = weibo.showUser(String.valueOf(accessToken.getUserId()));
					p.setFirstName(user.getScreenName());
				} catch (WeiboException e) {
					p.setFirstName(String.valueOf(accessToken.getUserId()));
					throw new SocialAuthException("Failed to get ScreenName.",e);
				}
				userProfile = p;
			}
		}
	
		return userProfile;
	}
	/**
	 * 授权验证通过，返回用户AccessToken
	 * @param requestToken
	 * @param verifier
	 * @return
	 */
	public AccessToken requstAccessToken(RequestToken requestToken,String verifier) throws Exception {
		AccessToken accessToken = null;
		try {
			System.setProperty("weibo4j.oauth.consumerKey", Weibo.CONSUMER_KEY);
			System.setProperty("weibo4j.oauth.consumerSecret",Weibo.CONSUMER_SECRET);

			Weibo weibo = new Weibo();
			accessToken = weibo.getOAuthAccessToken(requestToken.getToken(), requestToken.getTokenSecret(), verifier);
		} catch (Exception e) {
			LOG.error("Failed to get AccessToken.", e);
			throw new SocialAuthException("Failed to get AccessToken.",e);
		}
		return accessToken;
	}
}
