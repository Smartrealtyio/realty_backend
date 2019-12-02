(function($) {

    $.fn.validator = function() {
        $(this).each(function() {
            const input = $(this);
            input.on("input keydown keyup mousedown mouseup select contextmenu drop", function(event) {

            });
        });
    };


    let myMap, resultItem, resultsBlock, clusterer, searchResult = [];
    let mapCenter, placemark;

    let searchFormRequest = {};
    const calculateFormRequest = {};

    $(() => {
        const header = $('#header');
        const checkScroll = () => {
            const scrolled = window.pageYOffset || document.documentElement.scrollTop;
            if (header.is('.scrolled') && (scrolled < 50)) {
                header.removeClass('scrolled');
            } else if (!header.is('.scrolled') && (scrolled > 50)) {
                header.addClass('scrolled');
            }
        };
        window.onscroll = () => {
            checkScroll();
        };
        checkScroll();
        resultItem = $($('#result-item-tpl').html());
        resultsBlock = $('#visible-results');
    });


    // Инициализация карты
    const init = () => {
        mapCenter = [55.76, 37.64];
        myMap = new window['ymaps'].Map("realty-map", {
            center: mapCenter,
            zoom: 15
        });
    };
    window['ymaps'].ready(init);

    // Вычисление времени до метро
    const checkMetroDuration = () => {
        window['ymaps'].geocode(mapCenter, { kind: 'metro' }).then((res) => {
            const nearest = res.geoObjects.get(0);
            const multiRouteModel = new window['ymaps'].multiRouter.MultiRouteModel([
                mapCenter,
                nearest.geometry.getCoordinates()
            ], {
                routingMode: 'pedestrian'
            });
            multiRouteModel.events.add("requestsuccess", () => {
                const routes = multiRouteModel.getRoutes();
                calculateFormRequest.time_to_metro = Math.round(routes[0].properties.get("duration").value / 60);
            });
        });
    };

    // Установка маркера для формы расчета стоимости
    const setMarkerPosition = (coo) => {
        placemark.geometry.setCoordinates(coo);
        calculateFormRequest.lat = coo[0];
        calculateFormRequest.lng = coo[1];
        checkMetroDuration()
    };

    // Изменение координат маркера
    const iniMarkerPosition = (event) => {
        setMarkerPosition(event.get('coords'));
    };

    // Инициализация формы поиска выгодных предложений
    const iniSearchForm = () => {
        myMap.events.remove('click', iniMarkerPosition);
        myMap.geoObjects.remove(placemark);
        clusterer ? myMap.geoObjects.add(clusterer) : clusterer;
    };

    // Инициализация формы расчета стоимости
    const iniAnalyzeForm = () => {
        placemark = new window['ymaps'].Placemark(mapCenter, {}, {
            preset: 'islands#redIcon'
        });
        myMap.geoObjects.add(placemark);
        setMarkerPosition(mapCenter);
        myMap.events.add('click', iniMarkerPosition);
        clusterer ? myMap.geoObjects.remove(clusterer) : false;
    };


    const showResultPage = function(searchResult) {

        searchResult.flats.forEach((oneResultItem) => {

            const resultItemElement = resultItem.clone();
            oneResultItem.price_per_m = Math.round((oneResultItem.price / oneResultItem.full_sq));

            if (oneResultItem.metros && oneResultItem.metros.length) {
                oneResultItem.metro = oneResultItem.metros[0].station;
                oneResultItem.time_to_metro = oneResultItem.metros[0].time_to_metro;
            } else {
                oneResultItem.metro = undefined;
                oneResultItem.time_to_metro = undefined;
                oneResultItem.metro_hidden = "true";
            }

            for (let param in oneResultItem) {
                let visParam;
                if (param === 'price' || param === 'price_per_m') {
                    visParam = 'visible_' + param;
                } else {
                    visParam = param;
                }
                const valElement = resultItemElement.is('[data-' + visParam + ']') ? resultItemElement : resultItemElement.find('[data-' + visParam + ']');

                if (valElement.length) {

                    if (param === 'price' || param === 'price_per_m') {
                        oneResultItem[visParam] = oneResultItem[param].toLocaleString();
                    }
                    switch (valElement.data(visParam)) {
                        case 'text':
                            valElement.text(oneResultItem[visParam]);
                            break;
                        case 'background':
                            valElement.css({
                                backgroundImage: 'url(' + oneResultItem[visParam] + ')'
                            });
                            break;
                        default:
                            valElement.attr(valElement.data(visParam), oneResultItem[visParam]);
                    }
                }
            }
            resultsBlock.append(resultItemElement);
        })
    };


    $(() => {
        const results = $('#results');
        const resultPrice = $('#result_price');
        const resultDuration = $('#result_duration');
        $('[data-validator]').validator();


        $('#search-init').on('click', () => {
            iniSearchForm();
        });
        $('#analyze-init').on('click', () => {
            iniAnalyzeForm();
        });




        const createClusters = (data) => {
            var customItemContentLayout = window['ymaps'].templateLayoutFactory.createClass(
                '<h2 class=ballon_header>{{ properties.balloonContentHeader|raw }}</h2>' +
                '<div class=ballon_body>{{ properties.balloonContentBody|raw }}</div>' +
                '<div class=ballon_footer>{{ properties.balloonContentFooter|raw }}</div>'
            );

            clusterer = new window['ymaps'].Clusterer({
                clusterIcons: [
                    {
                        href: 'static/images/icons/pin_maps.svg',
                        size: [64, 64],
                        offset: [-32, -20]
                    }
                ],
                clusterDisableClickZoom: true,
                clusterOpenBalloonOnClick: true,
                clusterBalloonPanelMaxMapArea: 0,
                clusterBalloonContentLayoutWidth: 500,
                clusterBalloonItemContentLayout: customItemContentLayout,
                clusterBalloonLeftColumnWidth: 220
            });

            myMap.geoObjects.add(clusterer);

            const placemarks = [];

            for (let k in data) {

                const resultItem = data[k];

                const pm = new window['ymaps'].Placemark([resultItem.latitude, resultItem.longitude], {
                    balloonContentHeader: data[k].address,
                    balloonContentBody:
                        data[k].price.toLocaleString() + ' ₽<br/>' +
                        '<a href="' + data[k].link + '" target="_blank">' + data[k].link + '</a>'
                    ,
                    balloonContentFooter: '<small style="color: #ddd">Smart Realty</small>'
                }, {
                    iconLayout: 'default#imageWithContent',
                    iconImageHref: 'static/images/icons/pin_maps.svg',
                    iconImageSize: [56, 56],
                    iconImageOffset: [-28, -28],
                    iconContentOffset: [15, 15]
                });

                resultItem.placemark = pm;
                placemarks.push(pm);
            }

            clusterer.add(placemarks);
        };

        const createSearchRequest = () => {
            const fields = $('input, select', searchForm);
            const values = {};
            fields.each(function() {
                const itemModel = $(this);
                const itemType = itemModel.attr('type');
                if (itemModel.attr('name')) {
                    switch (itemType) {
                        case 'radio':
                            values[itemModel.attr('name')] = itemModel.is(':checked') ? itemModel.val() : values[itemModel.attr('name')] || undefined;
                            break;
                        case 'checkbox':
                            values[itemModel.attr('name')] = itemModel.is(':checked') ? '1' : '0';
                            break;
                        default:
                            values[itemModel.attr('name')] = itemModel.val() || undefined;
                    }
                }
            });
            const convertedFormData = values;
            for (const k in convertedFormData) {
                searchFormRequest[k] = convertedFormData[k];
            }

            const mapInfo = myMap.getBounds();
            searchFormRequest['latitude_from'] = mapInfo[0][0];
            searchFormRequest['latitude_to'] = mapInfo[1][0];
            searchFormRequest['longitude_from'] = mapInfo[0][1];
            searchFormRequest['longitude_to'] = mapInfo[1][1];
        };

        const createNavigation = (pagesCount) => {
            let navButtons = [], activeNavButton;

            const navigate = $('#results-navigate');
            navigate.html('');

            for (let n = 0; n < pagesCount; n++) {
                const navButton = $('<a>');
                navButton.attr('href', '#results-list');
                navButton.addClass('results-list_navigation-btn');
                navButton.text(n + 1);
                navButton.data('page', n);

                if (!n) {
                    activeNavButton = navButton.addClass('active');
                }

                navButton.on('click', function() {
                    activeNavButton.removeClass('active');
                    activeNavButton = $(this).addClass('active');
                    sendSearchForm(true, {
                        page: n + 1
                    });
                });
                navButtons.push(navButton);
                navigate.append(navButton);
            }
        };

        const iniShowMoreResults = () => {
            let currentPage = 1;
            document.getElementById('show-more-button').onclick = () => {
                currentPage++;
                sendSearchForm(true, {
                    page: currentPage
                });
            };
        };

        const sendSearchForm = (notCreateData, addData) => {

            const searchResultsBlock = $('#results-list-block');
            searchResultsBlock.addClass('in-progress');

            myMap.geoObjects.remove(clusterer);

            if (!notCreateData) {
                searchResultsBlock.hide();
                resultsBlock.html('');
                createSearchRequest();
            }

            let requestData = {...searchFormRequest};
            if (addData) {
                requestData = $.extend(requestData, addData);
            }


            $.ajax({
                url: '/api/mean/',
                // url: 'static/test.json',
                data: requestData
            }).then((res) => {
                searchResultsBlock.show().removeClass('in-progress');
                $('#search-items-count').text(res['count']);

                if (!notCreateData) {
                    searchResult = res['flats'];
                    iniShowMoreResults();
                    // createNavigation(res['max_page']);
                } else {
                    searchResult = searchResult.concat(res['flats']);
                }

                searchResultsBlock.removeClass('in-progress');
                if (res.flats.length) {
                    $('#visible-items-count').text(searchResult.length);
                    createClusters(searchResult);
                    showResultPage(res);
                }
            });
        };

        const searchForm = $('#search-form').on('submit', (event) => {
            event.preventDefault();
            sendSearchForm();
            return false;
        });




        const sendCalculateForm = () => {
            const fields = $('input, select', calculateForm);
            const values = {};

            fields.each(function() {
                const itemModel = $(this);
                const itemType = itemModel.attr('type');
                if (itemModel.attr('name')) {
                    switch (itemType) {
                        case 'checkbox':
                            values[itemModel.attr('name')] = itemModel.is(':checked') ? '1' : '0';
                            break;
                        default:
                            values[itemModel.attr('name')] = itemModel.val() || undefined;
                    }
                }
            });

            for (const k in values) {
                calculateFormRequest[k] = values[k];
            }

            results.hide();

            $.ajax({
                url: '/map',
                data: calculateFormRequest
            }).always(() => {
                results.show();
            }).then((res) => {
                resultPrice.text(res.Price.toLocaleString() + ' руб.');
                resultDuration.text(res.Duration + ' дн.');
            });
        };

        const calculateForm = $('#calculate-form').on('submit', (event) => {
            event.preventDefault();
            sendCalculateForm();
            return false;
        });

    });




})(jQuery);
